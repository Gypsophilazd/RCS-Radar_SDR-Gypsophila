"""
dsp_processor.py
================
Module 4a — DSP Processor
RCS-Radar-SDR  RM2026 Arena System

Streaming 4-RRC-FSK demodulation pipeline (fully stateful):

  IQ blocks → [Channeliser] → LPF → FM demod → RRC MF → AGC → Gardner TED
            → Slicer → Frame-sync state machine → PacketDecoder

ALL filters preserve cross-block state — this is critical for correct
streaming demodulation:

  • LPF    : scipy.signal.lfilter with zi initial-conditions vector
  • FM     : conjugate-delay discriminator; last IQ sample stored in
             self._prev_iq across consecutive blocks
  • RRC MF : overlap-save (state buffer prepended each block), equivalent
             to lfilter with zi but implemented with np.convolve so the
             same _make_rrc kernel used by the TX is reused verbatim

Previous bugs (all fixed here):
  ✗ LPF applied twice when channelize=True (once inside _apply_channeliser
    and once unconditionally in _process_block)
  ✗ np.convolve(..., mode="same") – no cross-block state
  ✗ FM demod reset iq_d[0] = iq[0] each block (should be self._prev_iq)
  ✗ RRC MF np.convolve(..., mode="full")[:len(fm)] – drops MF tail
  ✗ Channeliser phase discontinuity at block boundaries
  ✗ GUI IQ queue received bare array instead of (iq, centre_hz, sr_hz)
    tuple expected by visual_terminal.Dashboard._poll

Threading model
───────────────
  DSPProcessor.run_forever() blocks the calling thread.
  Call it in a daemon thread from main.py:

      t = threading.Thread(target=proc.run_forever, daemon=True)
      t.start()
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
from scipy.signal import firwin, lfilter

if TYPE_CHECKING:
    from config_manager import ConfigManager, FreqPlan

from packet_decoder import decode_frame

# ── DSP constants (must match TX chain) ───────────────────────────────────────
_BAUD_RATE       = 250_000
_FSK_DEV_HZ      = 250_000
_RRC_ALPHA       = 0.25
_RRC_SPAN        = 11
_AGC_TARGET      = float(np.sqrt(5.0))   # RMS of {-3,-1,+1,+3}: √((9+1+1+9)/4)=√5
_AGC_ALPHA       = 0.05
_LPF_TAPS        = 63                    # channeliser / pre-filter LPF tap count
_SUB_BLOCK_SYMS  = 2048                  # Gardner TED re-sync interval (symbols)

# ── Slicer thresholds for 4-FSK (boundaries between symbol levels) ────────────
_SLICER_THRESHOLDS = (-2.0, 0.0, 2.0)

# ── Maximum plausible DataLen (sanity-check against corrupted frames) ─────────
_MAX_DATA_LEN = 64   # CmdID(2) + max-payload(36) + CRC16(2) + margin


def _crc8_header(data: bytes) -> int:
    """CRC-8 (poly=0x31, init=0xFF) — RM2026 referee spec, over frame bytes[0:4]."""
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class DSPProcessor:
    """
    Consumes raw IQ blocks from *in_queue*, demodulates 4-RRC-FSK, and
    calls *on_frame* once per successfully decoded and CRC-verified packet.

    Parameters
    ----------
    config    : loaded ConfigManager
    in_queue  : queue.Queue[np.ndarray] fed by PlutoRxDriver (complex64)
    on_frame  : callback(dict) invoked in DSP thread for each decoded frame
    out_iq_q  : optional queue for raw IQ → GUI spectrum / waveform panels
                receives (iq, centre_hz, sr_hz) tuples
    out_sym_q : optional queue for recovered symbol values → GUI scatter panel
    """

    def __init__(
        self,
        config: "ConfigManager",
        in_queue: "queue.Queue[np.ndarray]",
        on_frame: Callable[[dict], None],
        out_iq_q:  Optional["queue.Queue"] = None,
        out_sym_q: Optional["queue.Queue[np.ndarray]"] = None,
    ):
        self._cfg      = config
        self._q_in     = in_queue
        self._on_frame = on_frame
        self._q_iq     = out_iq_q
        self._q_sym    = out_sym_q

        plan          = config.plan
        sr            = plan.sample_rate_hz
        self._sr      = sr
        self._sps     = sr // _BAUD_RATE

        # Centre frequency forwarded to GUI with each IQ block
        self._centre_hz  = float(plan.center_freq_hz)
        self._channelize = plan.channelize
        self._bc_offset  = plan.broadcast_offset_hz

        # ── Build filter kernels ──────────────────────────────────────────────
        self._rrc_kernel = _make_rrc(_RRC_ALPHA, _RRC_SPAN, self._sps)
        _lpf_cutoff      = plan.digital_lpf_cutoff_hz() if plan.channelize else 300_000.0
        self._lpf_kernel = firwin(_LPF_TAPS, _lpf_cutoff / (sr / 2.0),
                                  window="hamming").astype(np.float32)

        # ── Stateful filter initial conditions ────────────────────────────────
        # LPF: zi vector (length = taps - 1), complex because input is complex IQ
        self._lpf_zi   = np.zeros(_LPF_TAPS - 1, dtype=np.complex64)

        # FM demod: preserve last IQ sample across blocks
        self._prev_iq  = complex(1.0, 0.0)

        # RRC MF (overlap-save): state buffer of length (taps - 1)
        self._mf_state = np.zeros(len(self._rrc_kernel) - 1, dtype=np.float32)

        # Channeliser: continuous-phase frequency rotation via sample counter
        # (avoids float64 precision loss in long-running sessions)
        self._chan_n: int = 0

        # ── AGC ───────────────────────────────────────────────────────────────
        # Seed with target power so there is no large startup transient
        self._agc_power = _AGC_TARGET ** 2

        # ── Frame-sync state machine ──────────────────────────────────────────
        self._bit_sreg    : int       = 0
        self._frame_state : str       = "HUNT"
        self._frame_buf   : list[int] = []
        self._expected_body: int      = 0
        self._header_bits : int       = 0
        self._body_bits   : int       = 0

        self._stop_event = threading.Event()

    # ── public ───────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Block and process IQ blocks until stop() is called."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                iq = self._q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_block(iq)

    def stop(self) -> None:
        self._stop_event.set()

    # ── block pipeline ────────────────────────────────────────────────────────

    def _process_block(self, iq: np.ndarray) -> None:
        iq = np.asarray(iq, dtype=np.complex64)

        # Forward raw IQ to GUI spectrum / waveform panels (before any DSP)
        if self._q_iq is not None:
            try:
                self._q_iq.put_nowait((iq, self._centre_hz, self._sr))
            except queue.Full:
                pass

        # 1. Channeliser: phase-continuous shift of broadcast to DC
        #    Only applied when two distinct channels share the SDR band.
        if self._channelize:
            iq = self._apply_channeliser(iq)

        # 2. LPF — alias / jammer suppression  (STATEFUL via lfilter zi)
        #    Applied once here; NOT repeated inside _apply_channeliser.
        iq, self._lpf_zi = lfilter(self._lpf_kernel, [1.0], iq, zi=self._lpf_zi)

        # 3. FM discriminator — conjugate-delay  (STATEFUL via self._prev_iq)
        iq_prev    = np.empty_like(iq)
        iq_prev[0] = self._prev_iq
        iq_prev[1:] = iq[:-1]
        self._prev_iq = complex(iq[-1])          # saved for next block

        d_phase = np.angle(iq * np.conj(iq_prev))
        # rad/sample → normalised symbol level:  ÷ (2π·Δf/3 / sr) = ·sr/(2π·Δf/3)
        fm = (d_phase * (self._sr / (2.0 * np.pi)) / (_FSK_DEV_HZ / 3.0)
              ).astype(np.float32)

        # 4. RRC matched filter — overlap-save  (STATEFUL via self._mf_state)
        padded         = np.concatenate([self._mf_state, fm])
        mf             = np.convolve(padded, self._rrc_kernel, mode="valid")
        self._mf_state = padded[-(len(self._rrc_kernel) - 1):]

        if len(mf) == 0:
            return

        # 5. AGC — exponential-average, forward normalisation
        power = float(np.mean(mf ** 2))
        if power > 1e-12:
            self._agc_power = (1 - _AGC_ALPHA) * self._agc_power + _AGC_ALPHA * power
        mf = mf * (_AGC_TARGET / max(np.sqrt(self._agc_power), 1e-9))

        # 6. Gardner TED — blind symbol timing recovery
        symbols = self._gardner_ted(mf)

        # Forward symbol values to GUI scatter panel
        if self._q_sym is not None and len(symbols):
            try:
                self._q_sym.put_nowait(symbols.astype(np.float32))
            except queue.Full:
                pass

        # 7. Slicer + frame-sync state machine + decode
        for sym in symbols:
            for bit in _sym_to_bits(sym):
                self._push_bit(bit)

    # ── channeliser ──────────────────────────────────────────────────────────

    def _apply_channeliser(self, iq: np.ndarray) -> np.ndarray:
        """
        Phase-continuous frequency shift to move the broadcast carrier to DC.

        Uses a running sample counter (self._chan_n) so consecutive blocks
        rotate without any phase discontinuity at boundaries.  The integer
        counter wraps the modulo periodically to prevent float64 overflow
        in very long sessions.
        """
        n  = len(iq)
        ns = np.arange(self._chan_n, self._chan_n + n, dtype=np.float64)
        self._chan_n += n
        # Keep counter bounded to prevent float64 underflow (~2^53 samples ≈ 14 years)
        if self._chan_n > (1 << 50):
            self._chan_n = 0
        shift = np.exp(-1j * 2.0 * np.pi * self._bc_offset * ns / self._sr
                       ).astype(np.complex64)
        return iq * shift

    # ── Gardner TED ──────────────────────────────────────────────────────────

    def _gardner_ted(self, mf: np.ndarray) -> np.ndarray:
        """
        Blind-phase symbol timing recovery with sub-block re-search.

        Every _SUB_BLOCK_SYMS symbols the optimal sampling phase is
        re-evaluated by scanning all SPS candidate phases and selecting the
        one with maximum mean |energy|.  This works as a bang-bang TED:
        no PLL, no VCO — just periodic phase correction.

        Coverage: 2048 syms × SPS samples × 45 ppm ≈ 0.7 samples drift
        per sub-block, well within the ±0.5-sample basin of attraction.
        """
        sps    = self._sps
        n      = len(mf)
        symbols: list[float] = []

        resync_samples = _SUB_BLOCK_SYMS * sps
        pos = 0
        while pos + sps <= n:
            block_end = min(pos + resync_samples, n - sps)
            phase     = _best_phase(mf[pos:block_end], sps) if block_end > pos else 0
            idx = pos + phase
            while idx < n:
                symbols.append(float(mf[idx]))
                idx += sps
                if (idx - pos - phase) >= resync_samples:
                    break
            pos = idx

        return np.array(symbols, dtype=np.float32)

    # ── Frame-sync state machine ──────────────────────────────────────────────

    def _push_bit(self, bit: int) -> None:
        """
        Feed one bit through the SOF-hunting state machine.

        States
        ──────
        HUNT   : slide 8-bit shift register; trigger on SOF = 0xA5
        HEADER : collect next 4 bytes  (DataLen[2 LE] + Seq[1] + CRC8[1])
        BODY   : collect DataLen + 2 bytes (CmdID + Payload + CRC16)
                 then call decode_frame(); always return to HUNT afterwards

        Sanity guard: if DataLen > _MAX_DATA_LEN the header is treated as
        noise and the machine immediately returns to HUNT.
        """
        self._bit_sreg = ((self._bit_sreg << 1) | (bit & 1)) & 0xFF

        if self._frame_state == "HUNT":
            if self._bit_sreg == 0xA5:
                self._frame_buf   = [0xA5]
                self._header_bits = 0
                self._frame_state = "HEADER"

        elif self._frame_state == "HEADER":
            self._header_bits += 1
            if self._header_bits % 8 == 0:
                self._frame_buf.append(self._bit_sreg)
            if self._header_bits == 32:   # DataLen(2) + Seq(1) + CRC8(1) collected
                # 1. Header CRC8 check (bytes[0:4] → bytes[4])
                if _crc8_header(bytes(self._frame_buf[:4])) != self._frame_buf[4]:
                    self._frame_state = "HUNT"
                    return
                # 2. Plausibility check on DataLen
                data_len = self._frame_buf[1] | (self._frame_buf[2] << 8)
                if data_len == 0 or data_len > _MAX_DATA_LEN:
                    self._frame_state = "HUNT"
                    return
                # body = DataLen bytes (CmdID + Payload) + 2 bytes CRC16
                self._expected_body = data_len + 2
                self._body_bits     = 0
                self._frame_state   = "BODY"

        elif self._frame_state == "BODY":
            self._body_bits += 1
            if self._body_bits % 8 == 0:
                self._frame_buf.append(self._bit_sreg)
            if self._body_bits == self._expected_body * 8:
                result = decode_frame(bytes(self._frame_buf))
                if result is not None:
                    self._on_frame(result)
                self._frame_state = "HUNT"


# ─── Helper functions (module-level) ──────────────────────────────────────────

def _best_phase(block: np.ndarray, sps: int) -> int:
    """Return phase offset 0..sps-1 that maximises the mean |sample|² energy."""
    if len(block) < sps:
        return 0
    return int(np.argmax([float(np.mean(np.abs(block[p::sps]) ** 2))
                          for p in range(sps)]))


def _sym_to_bits(sym: float) -> list[int]:
    """Slice a real symbol value → 2-bit pair [msb, lsb]."""
    if   sym < _SLICER_THRESHOLDS[0]: dibit = 0b00
    elif sym < _SLICER_THRESHOLDS[1]: dibit = 0b01
    elif sym < _SLICER_THRESHOLDS[2]: dibit = 0b10
    else:                              dibit = 0b11
    return [(dibit >> 1) & 1, dibit & 1]


def _make_rrc(alpha: float, span: int, sps: int) -> np.ndarray:
    """Root Raised Cosine FIR kernel — span × sps taps, unit energy."""
    n_taps = span * sps
    t      = (np.arange(n_taps) - n_taps // 2) / sps
    h      = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + alpha * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-6:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num   = (np.sin(np.pi * ti * (1 - alpha))
                     + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha)))
            denom = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i]  = num / denom
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)
