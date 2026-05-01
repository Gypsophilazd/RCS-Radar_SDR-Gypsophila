"""
phy/clock_recovery.py
=====================
BlockPhaseClockRecovery — sub-block blind phase timing recovery.

Abstracts the Gardner / best-phase core logic from dsp_processor.py
into a reusable module with pluggable scoring functions.

Algorithm
─────────
The input sample stream is split into sub-blocks of sub_block_syms *
sps samples.  Within each sub-block all sps candidate timing phases
are evaluated via a configurable score function.  The phase with the
best score is selected, and symbol samples are extracted at that phase
offset.  This repeats for each sub-block, providing periodic phase
correction without a PLL/VCO.

score_mode
──────────
- "fsk4_energy"  : mean(abs(y)**2)       — legacy 4-RRC-FSK scoring
- "gfsk2_variance": var(y - median(y))    — 2-GFSK scoring (penalises
                    constant-envelope; rewards signal-like variation)

Recommended 2-GFSK defaults: sps=52, sub_block_syms=128,
score_mode="gfsk2_variance".
"""

from __future__ import annotations

import numpy as np


class BlockPhaseClockRecovery:
    """
    Blind-phase symbol timing recovery with periodic sub-block re-search.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    sub_block_syms : int
        Number of symbols between phase re-search.  At 45 ppm worst-case
        drift with sps=52, 128 symbols gives ~0.3 samples drift per
        sub-block, well within the ±sps/2 basin of attraction.
    score_mode : str
        "fsk4_energy" or "gfsk2_variance".
    """

    def __init__(
        self,
        sps: int,
        sub_block_syms: int = 128,
        score_mode: str = "gfsk2_variance",
    ):
        if score_mode not in ("fsk4_energy", "gfsk2_variance"):
            raise ValueError(
                f"score_mode must be 'fsk4_energy' or 'gfsk2_variance', "
                f"got {score_mode!r}"
            )
        self._sps = int(sps)
        self._sub_block_syms = int(sub_block_syms)
        self._score_mode = score_mode
        # Sub-block size in samples
        self._sub_samp = self._sub_block_syms * self._sps

    # ── public ──────────────────────────────────────────────────────────────

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract symbols from continuous samples using blind phase recovery.

        Sub-block phase re-search preserves symbol-grid continuity:
        after sampling a sub-block the next sample index (idx) is used
        as the start of the following sub-block, so the symbol grid
        continues uninterrupted across phase re-search points.
        """
        samples = np.asarray(samples, dtype=np.float32)
        n = len(samples)
        if n < self._sps:
            return np.array([], dtype=np.float32)

        symbols: list[float] = []
        pos = 0
        while pos + self._sps <= n:
            block_end = min(pos + self._sub_samp, n)
            if block_end <= pos:
                break
            block = samples[pos:block_end]
            phase = self._best_phase(block)
            idx = pos + phase
            while idx < block_end:
                symbols.append(float(samples[idx]))
                idx += self._sps
            # Continue from where we left off so the symbol grid is unbroken
            pos = idx

        return np.array(symbols, dtype=np.float32)

    # ── private ─────────────────────────────────────────────────────────────

    def _best_phase(self, block: np.ndarray) -> int:
        """Return phase 0..sps-1 that maximises the scoring function."""
        if len(block) < self._sps:
            return 0
        sps = self._sps
        scores = np.zeros(sps, dtype=np.float64)
        for p in range(sps):
            pts = block[p::sps]
            if len(pts) == 0:
                continue
            if self._score_mode == "fsk4_energy":
                scores[p] = float(np.mean(np.abs(pts) ** 2))
            elif self._score_mode == "gfsk2_variance":
                scores[p] = float(np.var(pts - np.median(pts)))
        return int(np.argmax(scores))
