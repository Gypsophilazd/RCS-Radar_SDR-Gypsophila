"""
phy/legacy_4rrcfsk.py
=====================
Legacy4RRCFSKModem — stateful streaming 4-RRC-FSK demodulator.

Preserves the original DSP chain from dsp_processor.py for regression
testing and compatibility with pre-RM2026 hardware configurations.

Pipeline
────────
  IQ blocks → [Channeliser] → LPF → FM discriminator → RRC MF → AGC →
  BlockPhaseClockRecovery → 4-level slicer → 2-bit symbols → bits

This modem outputs raw bits; frame synchronisation is handled by the
existing FrameSync state machine in dsp_processor.py.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from phy.filters import make_rrc
from phy.clock_recovery import BlockPhaseClockRecovery

# ── Legacy 4-FSK constants ──────────────────────────────────────────────────
_FSK_DEV_HZ     = 250_000
_RRC_ALPHA      = 0.25
_RRC_SPAN       = 11
_AGC_TARGET     = float(np.sqrt(5.0))   # RMS of {-3,-1,+1,+3}
_AGC_ALPHA      = 0.05
_LPF_TAPS       = 63
_SUB_BLOCK_SYMS = 2048

# 4-level slicer thresholds (midpoints between -3,-1,+1,+3)
_SLICER_THRESHOLDS = (-2.0, 0.0, 2.0)
# Dibit mapping: -3→00, -1→01, +1→10, +3→11
_SYM_TO_BITS = {-3: (0, 0), -1: (0, 1), 1: (1, 0), 3: (1, 1)}


class Legacy4RRCFSKModem:
    """
    Stateful streaming 4-RRC-FSK demodulator — exact preservation of the
    original dsp_processor.py chain.

    Parameters
    ----------
    sample_rate : float
        ADC sample rate in Hz.
    baude_rate : int
        Symbol baud rate (default 250_000).
    channelize : bool
        Enable channeliser (phase-continuous frequency shift).
    broadcast_offset_hz : float
        Offset for channeliser in Hz.
    """

    def __init__(
        self,
        sample_rate: float,
        baud_rate: int = 250_000,
        channelize: bool = False,
        broadcast_offset_hz: float = 0.0,
    ):
        self._sr = float(sample_rate)
        self._sps = int(sample_rate) // baud_rate
        self._channelize = channelize
        self._bc_offset = broadcast_offset_hz

        # ── RRC kernel ─────────────────────────────────────────────────────
        self._rrc_kernel = make_rrc(_RRC_ALPHA, _RRC_SPAN, self._sps)

        # ── LPF ────────────────────────────────────────────────────────────
        lpf_cutoff = 300_000.0
        self._lpf_kernel = firwin(
            _LPF_TAPS, lpf_cutoff / (sample_rate / 2.0), window="hamming"
        ).astype(np.float32)
        self._lpf_zi = np.zeros(_LPF_TAPS - 1, dtype=np.complex64)

        # ── RRC matched filter state ───────────────────────────────────────
        self._mf_state = np.zeros(len(self._rrc_kernel) - 1, dtype=np.float32)

        # ── Clock recovery ─────────────────────────────────────────────────
        self._clock = BlockPhaseClockRecovery(
            sps=self._sps,
            sub_block_syms=_SUB_BLOCK_SYMS,
            score_mode="fsk4_energy",
        )

        # ── FM discriminator state ─────────────────────────────────────────
        self._prev_iq = complex(1.0, 0.0)

        # ── Channeliser state ──────────────────────────────────────────────
        self._chan_n: int = 0

        # ── AGC state ──────────────────────────────────────────────────────
        self._agc_power = _AGC_TARGET ** 2

    # ── public ──────────────────────────────────────────────────────────────

    def push_iq(self, iq: np.ndarray) -> list[int]:
        """
        Demodulate a block of IQ samples into bits (2 bits per symbol).

        Returns
        -------
        list[int]
            Demodulated bits (0/1).
        """
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return []

        # 1. Optional channeliser
        if self._channelize:
            iq = self._apply_channeliser(iq)

        # 2. LPF (STATEFUL)
        iq, self._lpf_zi = lfilter(
            self._lpf_kernel, [1.0], iq, zi=self._lpf_zi
        )
        iq = np.asarray(iq, dtype=np.complex64)

        # 3. FM discriminator (STATEFUL)
        iq_prev = np.empty_like(iq)
        iq_prev[0] = self._prev_iq
        iq_prev[1:] = iq[:-1]
        self._prev_iq = complex(iq[-1])

        d_phase = np.angle(iq * np.conj(iq_prev))
        fm = (d_phase * (self._sr / (2.0 * np.pi))
              / (_FSK_DEV_HZ / 3.0)).astype(np.float32)

        # 4. RRC matched filter — overlap-save (STATEFUL)
        padded = np.concatenate([self._mf_state, fm])
        mf = np.convolve(padded, self._rrc_kernel, mode="valid")
        self._mf_state = padded[-(len(self._rrc_kernel) - 1):]
        if len(mf) == 0:
            return []

        # 5. AGC
        power = float(np.mean(mf ** 2))
        if power > 1e-12:
            self._agc_power = ((1 - _AGC_ALPHA) * self._agc_power
                               + _AGC_ALPHA * power)
        mf = mf * (_AGC_TARGET / max(np.sqrt(self._agc_power), 1e-9))

        # 6. Clock recovery → symbols
        symbols = self._clock.process(mf)
        if len(symbols) == 0:
            return []

        # 7. 4-level slicer → bits (2 bits per symbol)
        bits: list[int] = []
        for sym in symbols:
            bits.extend(self._slice_symbol(sym))
        return bits

    # ── private ─────────────────────────────────────────────────────────────

    def _apply_channeliser(self, iq: np.ndarray) -> np.ndarray:
        """Phase-continuous frequency shift to DC."""
        n = len(iq)
        ns = np.arange(self._chan_n, self._chan_n + n, dtype=np.float64)
        self._chan_n += n
        if self._chan_n > (1 << 50):
            self._chan_n = 0
        shift = np.exp(
            -1j * 2.0 * np.pi * self._bc_offset * ns / self._sr
        ).astype(np.complex64)
        return iq * shift

    @staticmethod
    def _slice_symbol(sym: float) -> list[int]:
        """4-level slicer: sym → 2-bit pair [msb, lsb]."""
        if sym < _SLICER_THRESHOLDS[0]:
            dibit = 0b00
        elif sym < _SLICER_THRESHOLDS[1]:
            dibit = 0b01
        elif sym < _SLICER_THRESHOLDS[2]:
            dibit = 0b10
        else:
            dibit = 0b11
        return [(dibit >> 1) & 1, dibit & 1]
