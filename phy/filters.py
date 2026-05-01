"""
phy/filters.py
==============
Filter design utilities for PHY-layer DSP.

- make_gaussian_filter — Gaussian pulse-shaping filter for GFSK
- make_rrc            — Root Raised Cosine filter (legacy 4-RRC-FSK)
"""

import numpy as np


def make_gaussian_filter(bt: float, span: int, sps: int) -> np.ndarray:
    """
    Gaussian low-pass filter for GFSK pulse shaping.

    The Gaussian frequency response:  H(f) = exp(-ln(2)/2 * (f/B)^2)
    where B = BT * symbol_rate is the 3-dB bandwidth.

    Impulse response:  h(t) = sqrt(2π/ln(2)) * B * exp(-2 (π B t)^2 / ln(2))

    Equivalent to a zero-mean Gaussian PDF with:
        σ = sqrt(ln(2)) / (2π * B)
      h(t) = (1 / sqrt(2π σ^2)) * exp(-t^2 / (2 σ^2))

    Parameters
    ----------
    bt : float
        Bandwidth-time product (e.g. 0.35 for GFSK BT=0.35).
    span : int
        Filter span in symbol periods.
    sps : int
        Samples per symbol.

    Returns
    -------
    np.ndarray
        Unit-energy normalised FIR coefficients (float32).
    """
    # 3-dB bandwidth in Hz (normalised to symbol rate = 1)
    B = float(bt)
    # σ of the Gaussian impulse response
    sigma = np.sqrt(np.log(2)) / (2.0 * np.pi * B)
    # Time vector: span * sps + 1 taps (odd length for symmetric zero-centre)
    n_taps = span * sps + 1
    t = (np.arange(n_taps) - (n_taps - 1) / 2) / sps  # normalised to symbol period

    # Gaussian impulse response
    h = np.exp(-t**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    # Normalise to unit peak so that an isolated +1 symbol convolved with this
    # filter produces a peak of exactly 1.0 at the symbol centre — the downstream
    # FM modulator then scales by deviation_hz to set the actual frequency swing.
    peak_idx = n_taps // 2
    h /= h[peak_idx]
    return h.astype(np.float32)


def make_rrc(alpha: float, span: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine FIR kernel.

    Parameters
    ----------
    alpha : float
        Roll-off factor (0 < alpha <= 1).
    span : int
        Filter span in symbol periods.
    sps : int
        Samples per symbol.

    Returns
    -------
    np.ndarray
        Unit-energy normalised FIR coefficients (float32).
    """
    n_taps = span * sps
    t = (np.arange(n_taps) - n_taps // 2) / sps
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + alpha * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-6:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = (np.sin(np.pi * ti * (1 - alpha))
                   + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha)))
            denom = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / denom
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)
