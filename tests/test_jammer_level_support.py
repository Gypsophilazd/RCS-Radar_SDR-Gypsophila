"""
tests/test_jammer_level_support.py
====================================
Unit tests for jammer-level TX/RX support and gain profiling.
No SDR hardware required.
"""

import math
import pytest
from config_manager import (
    get_deviation_hz,
    get_broadcast_frequency,
    get_jammer_frequency,
    get_sensitivity,
    load_config,
)
from phy.rf_profiler import RfProfiler


# ── Deviation formula tests ──────────────────────────────────────────────────

_SR = 1_000_000.0

def test_broadcast_deviation():
    """Broadcast deviation ≈ 250.8 kHz."""
    d = get_deviation_hz("broadcast", sample_rate=_SR)
    expected = 1.5756 * _SR / (2 * math.pi)
    assert abs(d - expected) < 1.0
    assert 245_000 < d < 256_000


def test_jammer_L1_deviation():
    """L1 deviation ≈ 450.8 kHz."""
    d = get_deviation_hz("jammer", level=1, sample_rate=_SR)
    expected = 2.8323 * _SR / (2 * math.pi)
    assert abs(d - expected) < 1.0
    assert 445_000 < d < 456_000


def test_jammer_L2_deviation():
    """L2 deviation ≈ 410.8 kHz."""
    d = get_deviation_hz("jammer", level=2, sample_rate=_SR)
    expected = 2.5809 * _SR / (2 * math.pi)
    assert abs(d - expected) < 1.0
    assert 405_000 < d < 416_000


def test_jammer_L3_deviation():
    """L3 deviation ≈ 105.8 kHz."""
    d = get_deviation_hz("jammer", level=3, sample_rate=_SR)
    expected = 0.6646 * _SR / (2 * math.pi)
    assert abs(d - expected) < 1.0
    assert 100_000 < d < 111_000


# ── Frequency helpers ────────────────────────────────────────────────────────

def test_broadcast_frequencies():
    assert get_broadcast_frequency("red") == 433_200_000.0
    assert get_broadcast_frequency("blue") == 433_920_000.0


def test_jammer_frequencies():
    assert get_jammer_frequency("red", 1) == 432_200_000.0
    assert get_jammer_frequency("red", 2) == 432_500_000.0
    assert get_jammer_frequency("red", 3) == 432_800_000.0
    assert get_jammer_frequency("blue", 1) == 434_920_000.0
    assert get_jammer_frequency("blue", 2) == 434_620_000.0
    assert get_jammer_frequency("blue", 3) == 434_320_000.0


# ── Sensitivity helpers ──────────────────────────────────────────────────────

def test_sensitivity_values():
    assert get_sensitivity("broadcast") == 1.5756
    assert get_sensitivity("jammer", 1) == 2.8323
    assert get_sensitivity("jammer", 2) == 2.5809
    assert get_sensitivity("jammer", 3) == 0.6646


# ── Config integration ───────────────────────────────────────────────────────

def test_phyconfig_has_both_deviations():
    """PhyConfig stores both broadcast and jammer deviations."""
    mgr = load_config()
    phy = mgr.phy_config
    assert phy.deviation_hz > 0
    assert phy.jammer_deviation_hz > 0
    # Broadcast deviation matches get_deviation_hz
    assert abs(phy.deviation_hz - get_deviation_hz("broadcast")) < 500.0


# ── RX source validation ─────────────────────────────────────────────────────

def test_rx_source_default_broadcast():
    mgr = load_config()
    assert mgr.rx_source == "broadcast"


def test_gain_mode_default_manual():
    mgr = load_config()
    assert mgr.gain_mode == "manual"


# ── RF Profiler tests ────────────────────────────────────────────────────────

class TestRfProfiler:
    def test_initial_state(self):
        p = RfProfiler(gain_db=30, gain_mode="manual")
        assert p.gain_db == 30
        assert p.gain_mode == "manual"
        assert p.iq_rms_mean == 0.0
        assert p.decoded_frames_per_sec == 0.0

    def test_update_tracks_rms(self):
        import numpy as np
        p = RfProfiler(gain_db=30)
        iq = np.ones(1000, dtype=np.complex64) * 0.3
        p.update(iq)
        assert 0.2 < p.iq_rms_mean < 0.4

    def test_semi_auto_reduces_gain_when_too_hot(self):
        import numpy as np
        p = RfProfiler(gain_db=30, gain_mode="semi_auto_guarded", window_s=0.0)
        iq = np.ones(1000, dtype=np.complex64) * 0.8  # > target_hi 0.30
        # 2-window confirmation required
        assert p.update(iq) is None  # window 1: pending
        delta = p.update(iq)         # window 2: confirmed
        assert delta == -5
        assert p.gain_db == 25

    def test_semi_auto_increases_gain_when_too_cold(self):
        import numpy as np
        p = RfProfiler(gain_db=30, gain_mode="semi_auto_guarded", window_s=0.0)
        iq = np.ones(1000, dtype=np.complex64) * 0.01  # < 0.02, frames_s=0
        assert p.update(iq) is None  # window 1
        delta = p.update(iq)         # window 2: confirmed
        assert delta == +5
        assert p.gain_db == 35

    def test_semi_auto_clamped(self):
        import numpy as np
        p = RfProfiler(gain_db=48, gain_mode="semi_auto_guarded", window_s=0.0,
                       gain_min_db=0, gain_max_db=50)
        iq = np.ones(1000, dtype=np.complex64) * 0.01
        assert p.update(iq) is None
        delta = p.update(iq)
        assert p.gain_db == 50  # 48+5 clamped

    def test_semi_auto_freeze_when_decoding(self):
        import numpy as np
        p = RfProfiler(gain_db=30, gain_mode="semi_auto_guarded", window_s=0.0)
        # Simulate reliable decoding
        p.record_decoded_frame()
        p.record_decoded_frame()
        iq = np.ones(1000, dtype=np.complex64) * 0.01
        delta = p.update(iq)
        # frames/s > 0.2 → freeze gain
        assert delta is None

    def test_record_crc_and_decode(self):
        p = RfProfiler(gain_db=30)
        p.record_crc(True)
        p.record_crc(True)
        p.record_crc(False)
        assert p.crc_pass_rate == 2.0 / 3.0

    def test_inter_decode_stats(self):
        p = RfProfiler(gain_db=30)
        # simulate decode timestamps: 0.0, 0.1, 0.25, 0.45
        import time as _time
        t0 = _time.monotonic()
        p._decode_times_s = [t0, t0 + 0.1, t0 + 0.25, t0 + 0.45]
        stats = p.inter_decode_stats()
        assert 0.09 < stats["median"] < 0.16   # intervals: 0.10, 0.15, 0.20 → median 0.15
        assert stats["p95"] is not None
        assert abs(stats["max"] - 0.20) < 1e-9

    def test_manual_mode_never_adjusts(self):
        import numpy as np
        p = RfProfiler(gain_db=30, gain_mode="manual", window_s=0.0)
        iq = np.ones(1000, dtype=np.complex64) * 0.8
        assert p.update(iq) is None
        assert p.gain_db == 30
