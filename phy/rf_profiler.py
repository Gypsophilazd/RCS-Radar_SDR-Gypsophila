"""
phy/rf_profiler.py
==================
Lightweight RF profiling and conservative semi-auto gain control.

Collects streaming RX statistics for manual inspection or automated
gain decisions.  Semi-auto mode adjusts gain conservatively: at most
one change per evaluation window, clamped to [0, 50] dB.

Usage
─────
    profiler = RfProfiler(gain_db=20, gain_mode="manual")
    profiler.update(iq_block)     # call once per IQ block
    print(profiler.summary())

    # In semi_auto mode the profiler returns a recommended gain delta
    # from update(); the caller applies it to the Pluto SDR.
"""

from __future__ import annotations

import time
from collections import deque


class RfProfiler:
    """
    Tracks per-block IQ RMS and per-second decode statistics.

    Parameters
    ----------
    gain_db : int
        Initial RX gain in dB.
    gain_mode : str
        "manual" (default) — no automatic gain changes.
        "semi_auto" — evaluate every 2-5 s, adjust if needed.
    window_s : float
        Evaluation window in seconds for semi_auto (default 3.0).
    iq_rms_target_lo : float
        Low threshold for IQ RMS (default 0.05).
    iq_rms_target_hi : float
        High threshold for IQ RMS (default 0.50).
    gain_step_db : int
        Gain adjustment step size (default 5 dB).
    gain_min_db : int
        Minimum allowed gain (default 0).
    gain_max_db : int
        Maximum allowed gain (default 50).
    """

    def __init__(
        self,
        gain_db: int = 20,
        gain_mode: str = "manual",
        window_s: float = 3.0,
        iq_rms_target_lo: float = 0.05,
        iq_rms_target_hi: float = 0.50,
        gain_step_db: int = 5,
        gain_min_db: int = 0,
        gain_max_db: int = 50,
    ):
        self._gain_db = gain_db
        self._gain_mode = gain_mode
        self._window_s = window_s
        self._target_lo = iq_rms_target_lo
        self._target_hi = iq_rms_target_hi
        self._gain_step = gain_step_db
        self._gain_min = gain_min_db
        self._gain_max = gain_max_db

        # IQ RMS stats over last N blocks
        self._iq_rms_samples: deque[float] = deque(maxlen=256)
        self._iq_rms_min: float = float("inf")
        self._iq_rms_max: float = float("-inf")

        # Per-second stats (counts since last reset)
        self._ac_hits: int = 0
        self._payloads: int = 0
        self._raw_frames: int = 0
        self._crc_pass: int = 0
        self._crc_fail: int = 0
        self._decoded_frames: int = 0
        self._decode_times_s: list[float] = []  # timestamps of decoded frames

        # Semi-auto state
        self._last_eval_time = time.monotonic()
        self._last_gain_change_time = 0.0
        self._evaluation_count = 0

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def gain_db(self) -> int:
        return self._gain_db

    @property
    def gain_mode(self) -> str:
        return self._gain_mode

    @property
    def iq_rms_mean(self) -> float:
        if not self._iq_rms_samples:
            return 0.0
        return sum(self._iq_rms_samples) / len(self._iq_rms_samples)

    @property
    def iq_rms_min(self) -> float:
        return self._iq_rms_min if self._iq_rms_min != float("inf") else 0.0

    @property
    def iq_rms_max(self) -> float:
        return self._iq_rms_max if self._iq_rms_max != float("-inf") else 0.0

    @property
    def ac_hits_per_sec(self) -> float:
        return self._per_sec(self._ac_hits)

    @property
    def payloads_per_sec(self) -> float:
        return self._per_sec(self._payloads)

    @property
    def raw_frames_per_sec(self) -> float:
        return self._per_sec(self._raw_frames)

    @property
    def crc_pass_rate(self) -> float:
        total = self._crc_pass + self._crc_fail
        return self._crc_pass / total if total > 0 else 0.0

    @property
    def decoded_frames_per_sec(self) -> float:
        return self._per_sec(self._decoded_frames)

    def inter_decode_stats(self) -> dict:
        """Return median / p95 / max inter-decode interval in seconds."""
        if len(self._decode_times_s) < 2:
            return {"median": None, "p95": None, "max": None}
        sorted_ts = sorted(self._decode_times_s)
        intervals = [sorted_ts[i] - sorted_ts[i - 1]
                     for i in range(1, len(sorted_ts))]
        intervals.sort()
        n = len(intervals)
        return {
            "median": intervals[n // 2],
            "p95": intervals[int(n * 0.95)],
            "max": intervals[-1],
        }

    def update(self, iq: "np.ndarray") -> int | None:
        """
        Process one IQ block.  Returns recommended gain delta in dB,
        or None if no change is needed / gain is manual.
        """
        import numpy as np
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        self._iq_rms_samples.append(rms)
        if rms < self._iq_rms_min:
            self._iq_rms_min = rms
        if rms > self._iq_rms_max:
            self._iq_rms_max = rms

        if self._gain_mode != "semi_auto":
            return None

        now = time.monotonic()
        if now - self._last_eval_time < self._window_s:
            return None

        self._last_eval_time = now
        self._evaluation_count += 1
        avg_rms = self.iq_rms_mean
        frames_s = self.decoded_frames_per_sec

        delta = 0
        if avg_rms > self._target_hi:
            delta = -self._gain_step
        elif avg_rms < self._target_lo and frames_s < 0.1:
            delta = +self._gain_step

        if delta == 0:
            return None

        new_gain = max(self._gain_min,
                       min(self._gain_max, self._gain_db + delta))
        actual_delta = new_gain - self._gain_db
        if actual_delta == 0:
            return None

        self._gain_db = new_gain
        self._last_gain_change_time = now
        print(f"[GAIN] semi_auto: RMS={avg_rms:.4f}  frames/s={frames_s:.1f}  "
              f"gain {self._gain_db - actual_delta}→{self._gain_db} dB  "
              f"(Δ={actual_delta:+d} dB)")
        return actual_delta

    def record_ac_hit(self) -> None:
        self._ac_hits += 1

    def record_payload(self) -> None:
        self._payloads += 1

    def record_raw_frame(self) -> None:
        self._raw_frames += 1

    def record_crc(self, passed: bool) -> None:
        if passed:
            self._crc_pass += 1
        else:
            self._crc_fail += 1

    def record_decoded_frame(self) -> None:
        self._decoded_frames += 1
        self._decode_times_s.append(time.monotonic())

    def summary(self) -> str:
        """One-line summary of current stats."""
        inter = self.inter_decode_stats()
        med = f"{inter['median']:.2f}s" if inter["median"] is not None else "n/a"
        return (
            f"[PROFILE] RMS={self.iq_rms_mean:.4f} "
            f"(min={self.iq_rms_min:.4f} max={self.iq_rms_max:.4f}) "
            f"AC/s={self.ac_hits_per_sec:.1f} "
            f"payload/s={self.payloads_per_sec:.1f} "
            f"frames/s={self.decoded_frames_per_sec:.1f} "
            f"CRC pass={self.crc_pass_rate:.2f} "
            f"inter-decode median={med} "
            f"gain={self._gain_db}dB [{self._gain_mode}]"
        )

    def reset_counts(self) -> None:
        """Reset per-second counters (call periodically for fresh rates)."""
        self._ac_hits = 0
        self._payloads = 0
        self._raw_frames = 0
        self._crc_pass = 0
        self._crc_fail = 0
        self._decoded_frames = 0
        # Keep decode_times for inter-decode stats but trim to last 100
        if len(self._decode_times_s) > 100:
            self._decode_times_s = self._decode_times_s[-100:]

    # ── private ─────────────────────────────────────────────────────────────

    def _per_sec(self, count: int) -> float:
        elapsed = time.monotonic() - self._last_eval_time
        return count / elapsed if elapsed > 0 else 0.0
