"""
phy/gfsk2_modem.py
==================
2-GFSK modem — stateful streaming demodulator and pure-software modulator.

GFSK2Demodulator: streaming demodulator.
  IQ blocks → [channelizer] → [LPF] → FM discriminator → normalize by
  deviation → DC removal → [Gaussian matched filter] →
  BlockPhaseClockRecovery → binary slicer → bits

gfsk2_modulate_bits: pure-software TX modulator for loopback tests.
  bits → 2-GFSK symbols → upsample → Gaussian pulse shape →
  FM integration → complex IQ

Key invariants (must NOT be violated):
  - 2-GFSK path never uses a 4-level slicer
  - 2-GFSK path outputs exactly 1 bit per symbol
  - FM discriminator preserves prev_iq across blocks
  - All filters preserve zi/state across blocks
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from phy.filters import make_gaussian_filter
from phy.clock_recovery import BlockPhaseClockRecovery


def gfsk2_modulate_bits(
    bits: list[int],
    sps: int = 52,
    bt: float = 0.35,
    span: int = 4,
    deviation_hz: float = 250_000.0,
    sample_rate: float = 1_000_000.0,
) -> np.ndarray:
    """
    Pure-software 2-GFSK modulator — converts bits to complex IQ samples.

    Modulation chain
    ────────────────
    1. Map each bit to symbol: 0 → -1, 1 → +1
    2. Upsample by sps (insert zeros between symbols)
    3. Convolve with Gaussian pulse-shaping filter (unit-peak normalised)
    4. Scale by deviation_hz to obtain instantaneous frequency in Hz
    5. Cumulative-sum (FM integration) → instantaneous phase
    6. Complex exponential → IQ samples

    Parameters
    ----------
    bits : list[int]
        Input bits (0/1).
    sps : int
        Samples per symbol (default 52).
    bt : float
        Gaussian bandwidth-time product (default 0.35).
    span : int
        Gaussian filter span in symbol periods (default 4).
    deviation_hz : float
        Peak frequency deviation in Hz.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        Complex64 IQ samples scaled to approximately ±0.5.
    """
    if len(bits) == 0:
        return np.array([], dtype=np.complex64)

    # 1. Bit → symbol mapping: 0→-1, 1→+1
    symbols = np.array([-1.0 if b == 0 else 1.0 for b in bits], dtype=np.float32)

    # 2. Upsample
    upsampled = np.zeros(len(symbols) * sps, dtype=np.float32)
    upsampled[::sps] = symbols

    # 3. Gaussian pulse shaping (unit-peak normalised filter)
    gauss_kernel = make_gaussian_filter(bt, span, sps)
    freq_pulse_full = np.convolve(upsampled, gauss_kernel, mode="full")

    # Trim convolution delay to align with upsample grid
    tx_delay = len(gauss_kernel) // 2
    freq_pulse = freq_pulse_full[tx_delay: tx_delay + len(upsampled)]

    # 4. Scale to instantaneous frequency (Hz)
    freq_hz = freq_pulse * deviation_hz

    # 5. FM integration → phase
    dt = 1.0 / sample_rate
    phase = np.cumsum(freq_hz) * 2.0 * np.pi * dt

    # 6. Complex IQ
    iq = np.exp(1j * phase).astype(np.complex64)
    iq *= 0.5
    return iq


class GFSK2Demodulator:
    """
    Stateful streaming 2-GFSK demodulator.

    By default operates with no LPF and no matched filter — the raw
    FM discriminator output is clean enough for symbol detection at
    moderate-to-high SNR.  Enable LPF / matched filter for noisy or
    heavily interfered channels.

    Parameters
    ----------
    sps : int
        Samples per symbol at demod_sample_rate (default 52).
    bt : float
        Gaussian bandwidth-time product (default 0.35).
    span : int
        Gaussian filter span in symbols (default 4).
    deviation_hz : float
        Peak frequency deviation in Hz.  Computed from sensitivity:
        deviation = sensitivity * sample_rate / (2*pi).
    sample_rate : float
        Demodulation sample rate in Hz — the rate at which FM discriminator
        and clock recovery operate (default 1_000_000).
    input_sample_rate : float
        ADC input sample rate in Hz.  When different from *sample_rate*,
        the modem applies a decimation stage after the channelizer.
        Default: same as *sample_rate*.
    channelizer_offset_hz : float or None
        If set, apply a phase-continuous frequency shift at the input
        sample rate BEFORE decimation.
    threshold_mode : str
        "zero" — slice at 0.0
        "running_median" — use running median of symbol values
    use_lpf : bool
        Enable pre-FM LPF (default False — clean signals don't need it).
    lpf_cutoff_hz : float
        LPF cutoff in Hz.  Default: deviation_hz * 1.5.
    lpf_taps : int
        Number of LPF FIR taps (default 63).
    use_matched_filter : bool
        Enable Gaussian matched filter after FM discriminator
        (default False — only needed for noisy channels).
    sub_block_syms : int
        Symbols between clock phase re-search (default 512).
    """

    def __init__(
        self,
        sps: int = 52,
        bt: float = 0.35,
        span: int = 4,
        deviation_hz: float = 250_000.0,
        sample_rate: float = 1_000_000.0,
        input_sample_rate: float | None = None,
        channelizer_offset_hz: float | None = None,
        threshold_mode: str = "zero",
        use_lpf: bool = False,
        lpf_cutoff_hz: float | None = None,
        lpf_taps: int = 63,
        use_matched_filter: bool = False,
        sub_block_syms: int = 512,
    ):
        self._sps = int(sps)
        self._bt = bt
        self._span = span
        self._deviation = float(deviation_hz)
        self._demod_sr = float(sample_rate)        # rate after decimation
        self._input_sr = float(input_sample_rate if input_sample_rate is not None
                               else sample_rate)
        self._chan_offset = channelizer_offset_hz
        self._threshold_mode = threshold_mode
        self._sub_block_syms = sub_block_syms
        self._use_lpf = use_lpf
        self._use_mf = use_matched_filter

        # ── Decimation ────────────────────────────────────────────────────────
        self._decim = round(self._input_sr / self._demod_sr)
        if self._decim < 1:
            raise ValueError(
                f"input_sample_rate ({self._input_sr}) must be >= "
                f"sample_rate ({self._demod_sr})"
            )
        self._needs_decim = self._decim > 1

        # Anti-aliasing LPF for decimation: cutoff at 40% of demod Nyquist
        # to allow for FIR transition band.  For GFSK with deviation ≤ 450 kHz
        # this still passes the full modulated bandwidth.
        if self._needs_decim:
            nyq_input = self._input_sr / 2.0
            cutoff_decim = self._demod_sr * 0.40
            self._decim_lpf = firwin(
                lpf_taps, cutoff_decim / nyq_input, window="hamming"
            ).astype(np.float32)
            self._decim_zi = np.zeros(lpf_taps - 1, dtype=np.complex64)
        else:
            self._decim_lpf = None
            self._decim_zi = None

        # ── LPF (optional, at demod rate) ─────────────────────────────────────
        if use_lpf:
            cutoff = lpf_cutoff_hz if lpf_cutoff_hz is not None else deviation_hz * 1.5
            nyq = self._demod_sr / 2.0
            self._lpf_kernel = firwin(
                lpf_taps, cutoff / nyq, window="hamming"
            ).astype(np.float32)
            self._lpf_zi = np.zeros(lpf_taps - 1, dtype=np.complex64)
        else:
            self._lpf_kernel = None
            self._lpf_zi = None

        # ── Gaussian matched filter (optional) ────────────────────────────────
        if use_matched_filter:
            self._gauss_kernel = make_gaussian_filter(bt, span, sps)
            self._gauss_zi = np.zeros(len(self._gauss_kernel) - 1, dtype=np.float32)
        else:
            self._gauss_kernel = None
            self._gauss_zi = None

        # ── Clock recovery ────────────────────────────────────────────────────
        self._clock = BlockPhaseClockRecovery(
            sps=sps,
            sub_block_syms=sub_block_syms,
            score_mode="gfsk2_variance",
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._prev_iq = complex(1.0, 0.0)
        self._chan_n: int = 0
        self._dc_est = 0.0         # running DC estimate for bias removal
        self._dc_alpha = 0.01      # EMA coefficient for DC tracking
        self._median_buf: list[float] = []

    # ── public ────────────────────────────────────────────────────────────────

    def push_iq(self, iq: np.ndarray) -> list[int]:
        """
        Demodulate a block of complex IQ samples into bits.

        When input_sample_rate > sample_rate, the modem channelises
        (if configured), low-pass filters, and decimates to the
        demodulation sample rate before FM discrimination.

        Parameters
        ----------
        iq : np.ndarray
            Complex64 IQ samples at input_sample_rate.

        Returns
        -------
        list[int]
            Demodulated bits (0/1), one bit per symbol.
        """
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return []

        # 1. Optional channelizer — phase-continuous frequency shift
        #    Applied at INPUT sample rate.
        if self._chan_offset is not None and abs(self._chan_offset) > 0.0:
            iq = self._apply_channelizer(iq)

        # 2. Decimation: anti-alias LPF + downsample to demod rate
        if self._needs_decim:
            iq, self._decim_zi = lfilter(
                self._decim_lpf, [1.0], iq, zi=self._decim_zi
            )
            iq = np.asarray(iq, dtype=np.complex64)
            iq = iq[::self._decim].copy()

        # 3. Optional LPF at demod rate
        if self._use_lpf:
            iq, self._lpf_zi = lfilter(
                self._lpf_kernel, [1.0], iq, zi=self._lpf_zi
            )
            iq = np.asarray(iq, dtype=np.complex64)

        # 4. FM discriminator — conjugate-delay (STATEFUL via prev_iq)
        #    Operates at _demod_sr.
        iq_prev = np.empty_like(iq)
        iq_prev[0] = self._prev_iq
        iq_prev[1:] = iq[:-1]
        self._prev_iq = complex(iq[-1])

        d_phase = np.angle(iq * np.conj(iq_prev))
        freq_hz = d_phase * (self._demod_sr / (2.0 * np.pi))

        # 5. Normalize by deviation (scale so symbol levels are ~±1)
        fm = (freq_hz / self._deviation).astype(np.float32)

        # 6. DC/bias removal — subtract running estimate
        block_dc = float(np.mean(fm))
        self._dc_est = ((1 - self._dc_alpha) * self._dc_est
                        + self._dc_alpha * block_dc)
        fm = fm - self._dc_est

        # 7. Optional Gaussian matched filter (STATEFUL, overlap-save)
        if self._use_mf:
            fm, self._gauss_zi = self._apply_gauss_filter(fm)

        # 8. Block phase clock recovery → symbols
        symbols = self._clock.process(fm)
        if len(symbols) == 0:
            return []

        # 9. Binary slicer → bits (1 symbol = 1 bit)
        threshold = self._get_threshold(symbols)
        bits = [1 if s > threshold else 0 for s in symbols]

        return bits

    # ── private ───────────────────────────────────────────────────────────────

    def _apply_channelizer(self, iq: np.ndarray) -> np.ndarray:
        """Phase-continuous frequency shift at INPUT sample rate."""
        n = len(iq)
        ns = np.arange(self._chan_n, self._chan_n + n, dtype=np.float64)
        self._chan_n += n
        if self._chan_n > (1 << 50):
            self._chan_n = 0
        offset = self._chan_offset
        shift = np.exp(
            -1j * 2.0 * np.pi * offset * ns / self._input_sr
        ).astype(np.complex64)
        return iq * shift

    def _apply_gauss_filter(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stateful Gaussian filter via overlap-save."""
        k = self._gauss_kernel
        padded = np.concatenate([self._gauss_zi, x])
        out = np.convolve(padded, k, mode="valid")
        new_zi = padded[-(len(k) - 1):]
        return out.astype(np.float32), new_zi.astype(np.float32)

    def _get_threshold(self, symbols: np.ndarray) -> float:
        """Compute slicing threshold based on threshold_mode."""
        if self._threshold_mode == "zero":
            return 0.0
        elif self._threshold_mode == "running_median":
            self._median_buf.extend(symbols.tolist())
            if len(self._median_buf) > 5000:
                self._median_buf = self._median_buf[-5000:]
            return float(np.median(self._median_buf)) if self._median_buf else 0.0
        else:
            return 0.0
