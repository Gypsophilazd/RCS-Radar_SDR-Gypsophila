"""
dsp_processor.py
================
Module 4a — DSP Processor
RCS-Radar-SDR  RM2026 Arena System

Streaming demodulation pipeline with mode dispatch:

  Mode "2gfsk" (default):
    IQ blocks → GFSK2Demodulator → AirPacketDeframer →
    PayloadStreamReassembler → decode_frame → callback

  Mode "4rrcfsk_legacy":
    IQ blocks → Legacy4RRCFSKModem → FrameSync → decode_frame → callback

All filters preserve cross-block state — critical for correct streaming
demodulation.

Threading model
───────────────
  DSPProcessor.run_forever() blocks the calling thread.
  Call it in a daemon thread from main.py.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config_manager import ConfigManager, FreqPlan

from packet_decoder import decode_frame, crc8_rm, verify_frame
from phy.gfsk2_modem import GFSK2Demodulator
from phy.air_packet import AirPacketDeframer
from phy.stream_reassembler import PayloadStreamReassembler
from phy.legacy_4rrcfsk import Legacy4RRCFSKModem
from phy.rf_profiler import RfProfiler

# ── Maximum plausible DataLen ────────────────────────────────────────────────
_MAX_DATA_LEN = 64   # CmdID(2) + max-payload(36) + CRC16(2) + margin


class DSPProcessor:
    """
    Consumes raw IQ blocks, demodulates, and calls *on_frame* for each
    decoded and CRC-verified RoboMaster frame.

    Parameters
    ----------
    config    : loaded ConfigManager
    in_queue  : queue.Queue[np.ndarray] fed by PlutoRxDriver (complex64)
    on_frame  : callback(dict) invoked in DSP thread for each decoded frame
    out_iq_q  : optional queue for raw IQ → GUI spectrum / waveform
    out_sym_q : optional queue for symbol values → GUI scatter panel
    rx_source : "broadcast" (default) or "jammer" — RX listen target
    direct_tune : bool
        If True, the SDR is directly tuned to the channel of interest
        at 1 MSPS — no channelizer/decimation needed regardless of
        plan.channelize.
    input_sample_rate_hz : float or None
        Override the SDR input sample rate.
    threshold_mode : str
        "zero" or "running_median".
    use_lpf : bool
        Enable pre-FM LPF.
    use_mf : bool
        Enable Gaussian matched filter.
    decim_cutoff_hz : float or None
        Custom decimation LPF cutoff (None = auto from deviation).
    """

    def __init__(
        self,
        config: "ConfigManager",
        in_queue: "queue.Queue[np.ndarray]",
        on_frame: Callable[[dict], None],
        out_iq_q:  Optional["queue.Queue"] = None,
        out_sym_q: Optional["queue.Queue[np.ndarray]"] = None,
        rx_source: str = "broadcast",
        direct_tune: bool = False,
        input_sample_rate_hz: float | None = None,
        threshold_mode: str = "zero",
        use_lpf: bool = False,
        use_mf: bool = False,
        decim_cutoff_hz: float | None = None,
    ):
        self._cfg      = config
        self._q_in     = in_queue
        self._on_frame = on_frame
        self._q_iq     = out_iq_q
        self._q_sym    = out_sym_q
        self._rx_source = rx_source

        plan          = config.plan
        phy           = config.phy_config
        self._sr      = plan.sample_rate_hz
        self._centre_hz = float(plan.center_freq_hz)
        self._phy_mode = phy.mode

        # ── Build modem chain by mode ───────────────────────────────────────
        if phy.mode == "2gfsk":
            # Select deviation and channelizer offset based on RX source
            # When direct_tune is True, the SDR is already tuned to the channel
            # at 1 MSPS — no channelization/decimation needed.
            if rx_source == "jammer":
                rx_dev = phy.jammer_deviation_hz
                chan_offset = (plan.jammer_offset_hz
                               if (plan.channelize and not direct_tune
                                   and plan.jammer_offset_hz != 0.0)
                               else None)
                ac_mode = "jammer"
            else:
                rx_dev = phy.deviation_hz
                chan_offset = (plan.broadcast_offset_hz
                               if (plan.channelize and not direct_tune)
                               else None)
                ac_mode = phy.access_code_mode

            # 2-GFSK chain: demod → deframer → reassembler → decode
            # input_sample_rate_hz override wins over plan.sample_rate_hz
            # (used when --rx-freq direct-tunes Pluto to 1 MSPS)
            input_sr = (input_sample_rate_hz
                        if input_sample_rate_hz is not None
                        else float(plan.sample_rate_hz))

            self._demod = GFSK2Demodulator(
                sps=phy.sps,
                bt=phy.bt,
                span=phy.span,
                deviation_hz=rx_dev,
                sample_rate=phy.sample_rate,
                input_sample_rate=input_sr,
                channelizer_offset_hz=chan_offset,
                threshold_mode=threshold_mode,
                use_lpf=use_lpf,
                use_matched_filter=use_mf,
                sub_block_syms=phy.sub_block_syms,
                decim_cutoff_hz=decim_cutoff_hz,
            )
            self._deframer = AirPacketDeframer(mode=ac_mode)
            self._reassembler = PayloadStreamReassembler()
            self._legacy_modem = None

            # ── Debug: print RX DSP configuration ───────────────────────────
            decim_factor = (input_sr / phy.sample_rate
                            if input_sr != phy.sample_rate else 1)
            print(f"[DSP] RX source      : {rx_source}")
            print(f"[DSP] SDR sample rate: {input_sr / 1e6:.2f} MSPS")
            print(f"[DSP] Demod rate     : {phy.sample_rate / 1e6:.2f} MSPS")
            print(f"[DSP] Deviation      : {rx_dev / 1e3:.1f} kHz")
            print(f"[DSP] Channelizer    : {chan_offset / 1e3:+.1f} kHz" if chan_offset
                  else "[DSP] Channelizer    : disabled")
            print(f"[DSP] Decimation     : {decim_factor:.0f}:1" if decim_factor > 1
                  else "[DSP] Decimation     : none (1:1)")
            if decim_factor > 1:
                print(f"[DSP] Decim cutoff   : {self._demod.decim_cutoff_hz / 1e3:.0f} kHz")
            print(f"[DSP] AC mode        : {ac_mode}")
            print(f"[DSP] Threshold      : {threshold_mode}")
            print(f"[DSP] LPF            : {'on' if use_lpf else 'off'}  "
                  f"MF: {'on' if use_mf else 'off'}")

            # ── Frame-sync state machine (unused in 2-GFSK mode) ────────────
            self._bit_sreg    : int       = 0
            self._frame_state : str       = "HUNT"
            self._frame_buf   : list[int] = []
            self._expected_body: int      = 0
            self._header_bits : int       = 0
            self._body_bits   : int       = 0

        elif phy.mode == "4rrcfsk_legacy":
            self._legacy_modem = Legacy4RRCFSKModem(
                sample_rate=self._sr,
                baud_rate=250_000,
                channelize=plan.channelize,
                broadcast_offset_hz=plan.broadcast_offset_hz,
            )
            self._demod = None
            self._deframer = None
            self._reassembler = None

            # ── Frame-sync state machine (used in legacy mode) ──────────────
            self._bit_sreg    : int       = 0
            self._frame_state : str       = "HUNT"
            self._frame_buf   : list[int] = []
            self._expected_body: int      = 0
            self._header_bits : int       = 0
            self._body_bits   : int       = 0
        else:
            raise ValueError(f"Unknown phy_mode: {phy.mode!r}")

        # ── RF profiler (semi_auto gain support) ─────────────────────────────
        self._profiler = RfProfiler(
            gain_db=config.rx_gain_db,
            gain_mode=config.gain_mode,
        )
        self._profiler_summary_blocks = 0

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

    # ── block pipeline ───────────────────────────────────────────────────────

    @property
    def profiler(self) -> RfProfiler:
        return self._profiler

    def _process_block(self, iq: np.ndarray) -> None:
        iq = np.asarray(iq, dtype=np.complex64)

        # Profiler: update IQ RMS stats, check for gain adjustments
        gain_delta = self._profiler.update(iq)

        # Forward raw IQ to GUI
        if self._q_iq is not None:
            try:
                self._q_iq.put_nowait((iq, self._centre_hz, self._sr))
            except queue.Full:
                pass

        if self._phy_mode == "2gfsk":
            self._process_block_2gfsk(iq)
        else:
            self._process_block_legacy(iq)

    # ── 2-GFSK path ──────────────────────────────────────────────────────────

    def _process_block_2gfsk(self, iq: np.ndarray) -> None:
        """2-GFSK: demodulate → deframe → reassemble → decode."""
        # 1. Demodulate IQ to bits
        bits = self._demod.push_iq(iq)

        # 2. Deframe: hunt access codes, extract 15-byte payloads
        payloads = self._deframer.push_bits(bits)

        # ── Post-channelizer diagnostics (every ~8 blocks) ─────────────────
        self._profiler_summary_blocks += 1
        if self._profiler_summary_blocks % 8 == 0:
            self._print_diag(iq, bits, payloads)

        # Track AC hits and payloads
        if payloads:
            self._profiler.record_ac_hit()
            self._profiler.record_payload()

        # 3. Reassemble payload stream into RM frames, decode, callback
        for payload in payloads:
            raw_frames = self._reassembler.push_payload(payload)
            for raw in raw_frames:
                self._profiler.record_raw_frame()
                crc_ok = verify_frame(raw)
                self._profiler.record_crc(crc_ok)
                if crc_ok:
                    result = decode_frame(raw)
                    if result is not None:
                        self._profiler.record_decoded_frame()
                        self._on_frame(result)

    def _print_diag(self, iq: np.ndarray, bits: list[int],
                    payloads: list[bytes]) -> None:
        """Print per-block diagnostic stats."""
        import numpy as np
        iq_rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        # Quick FM stats on raw IQ
        if len(iq) > 1:
            fm_q = np.angle(iq[1:] * np.conj(iq[:-1]))
            fm_hz = fm_q * (self._sr / (2.0 * np.pi))
            fm_mean = float(np.mean(fm_hz)) / 1e3
            fm_std = float(np.std(fm_hz)) / 1e3
        else:
            fm_mean = fm_std = 0.0
        n_bits = len(bits)
        n_ac = len(payloads)
        print(f"[DIAG] blk={self._profiler_summary_blocks}  "
              f"IQ_RMS={iq_rms:.4f}  "
              f"FM μ={fm_mean:+.1f}kHz σ={fm_std:.0f}kHz  "
              f"bits={n_bits}  AC/payloads={n_ac}  "
              f"frames={self._profiler.decoded_frames_per_sec:.1f}/s", flush=True)

        # Print profiler summary on the same cadence
        print(self._profiler.summary())
        self._profiler.reset_counts()

    # ── Legacy 4-RRC-FSK path ────────────────────────────────────────────────

    def _process_block_legacy(self, iq: np.ndarray) -> None:
        """Legacy 4-RRC-FSK: demodulate → FrameSync → decode."""
        bits = self._legacy_modem.push_iq(iq)

        if self._q_sym is not None and len(bits):
            try:
                self._q_sym.put_nowait(np.array(bits[-128:], dtype=np.float32))
            except queue.Full:
                pass

        for bit in bits:
            self._push_bit(bit)

    def _on_legacy_frame(self, result: dict) -> None:
        """Wrapper for legacy frame callback — records profiling stats."""
        self._profiler.record_raw_frame()
        self._profiler.record_crc(True)
        self._profiler.record_decoded_frame()
        self._on_frame(result)

    # ── Frame-sync state machine (legacy mode) ───────────────────────────────

    def _push_bit(self, bit: int) -> None:
        """
        Feed one bit through the SOF-hunting state machine.

        States: HUNT → HEADER → BODY → decode → HUNT
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
            if self._header_bits == 32:
                if crc8_rm(bytes(self._frame_buf[:4])) != self._frame_buf[4]:
                    self._frame_state = "HUNT"
                    return
                data_len = self._frame_buf[1] | (self._frame_buf[2] << 8)
                if data_len == 0 or data_len > _MAX_DATA_LEN:
                    self._frame_state = "HUNT"
                    return
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
                    self._on_legacy_frame(result)
                self._frame_state = "HUNT"
