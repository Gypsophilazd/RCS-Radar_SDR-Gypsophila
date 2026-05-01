"""
tests/test_rx_sample_rate_mismatch.py
======================================
Unit tests for RX sample-rate fix: decimation and --rx-freq.
No SDR hardware required.
"""

import struct
import math
import numpy as np
import pytest
from config_manager import load_config
from phy.gfsk2_modem import GFSK2Demodulator, gfsk2_modulate_bits
from phy.air_packet import AirPacketDeframer, ac_to_bits
from phy.stream_reassembler import PayloadStreamReassembler
from packet_decoder import crc8_rm, crc16_rm, SOF, decode_frame


_SPS = 52; _BT = 0.35; _SPAN = 4; _DEV = 250000.0; _DEMOD_SR = 1e6


def _build_test_bits() -> list[int]:
    """Build 216-bit air packet with a valid RM frame inside."""
    frame = _build_rm_frame(0x0A06, b"RM2026")
    assert len(frame) == 15
    ac_bits = ac_to_bits(0x2F6F4C74B914492E)
    hdr = struct.pack(">HH", 15, 15)
    hdr_bits = []
    for b in hdr:
        for i in range(7, -1, -1):
            hdr_bits.append((b >> i) & 1)
    payload_bits = []
    for b in frame:
        for i in range(7, -1, -1):
            payload_bits.append((b >> i) & 1)
    return ac_bits + hdr_bits + payload_bits


def _build_rm_frame(cmd_id, payload, seq=1):
    data_bytes = struct.pack("<H", cmd_id) + payload
    data_len = len(data_bytes)
    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, seq])
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + data_bytes
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


# ── Test: level 0 uses 1 MSPS ──────────────────────────────────────────────

def test_level_0_sample_rate():
    """target_jammer_level=0 → plan.sample_rate_hz == 1_000_000."""
    import json, tempfile, os
    from config_manager import ConfigManager
    cfg = {"team_color": "red", "target_jammer_level": 0, "phy_mode": "2gfsk"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        tmp = f.name
    try:
        mgr = ConfigManager(tmp).load()
        plan = mgr.plan
        assert plan.jammer_level == 0
        assert plan.sample_rate_hz == 1_000_000
        assert not plan.channelize
    finally:
        os.unlink(tmp)


# ── Test: equal rates → no decimation ──────────────────────────────────────

def test_demod_equal_rates():
    """When input_sample_rate == sample_rate, no decimation occurs."""
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV,
        sample_rate=1_000_000,
        input_sample_rate=1_000_000,
    )
    assert not demod._needs_decim
    assert demod._decim == 1


# ── Test: decimation 3:1 ──────────────────────────────────────────────────

def test_decimation_loopback_2to1():
    """2 MSPS input → decimate 2:1 → demodulate correctly at 1 MSPS."""
    tx_bits = _build_test_bits()

    # Modulate at 2 MSPS with SPS = 104 (2 × 52)
    iq_2msps = gfsk2_modulate_bits(
        tx_bits,
        sps=104, bt=_BT, span=_SPAN,
        deviation_hz=_DEV, sample_rate=2_000_000,
    )

    # Demodulate: input 2 MSPS → decimate 2:1 → 1 MSPS internal
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV,
        sample_rate=1_000_000,
        input_sample_rate=2_000_000,
    )
    assert demod._needs_decim
    assert demod._decim == 2

    bits = demod.push_iq(iq_2msps)

    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) >= 1, f"Decimation loopback failed: {len(payloads)} payloads"

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) >= 1
    decoded = decode_frame(raw_frames[0])
    assert decoded is not None
    assert decoded["key"] == "RM2026"


# ── Test: 1 MSPS no-decimal path still works ───────────────────────────────

def test_no_decimation_still_works():
    """Existing 1 MSPS path is unbroken."""
    tx_bits = _build_test_bits()
    iq = gfsk2_modulate_bits(
        tx_bits,
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV, sample_rate=_DEMOD_SR,
    )

    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV,
        sample_rate=1_000_000,
    )
    assert not demod._needs_decim

    bits = demod.push_iq(iq)
    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) >= 1

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    decoded = decode_frame(raw_frames[0])
    assert decoded is not None
    assert decoded["key"] == "RM2026"


# ── Test: decimation fails with inverted rates ─────────────────────────────

def test_inverted_rates_raises():
    """input_sample_rate < sample_rate should raise ValueError."""
    with pytest.raises(ValueError, match="input_sample_rate"):
        GFSK2Demodulator(
            sps=_SPS, bt=_BT, span=_SPAN,
            deviation_hz=_DEV,
            sample_rate=3_000_000,
            input_sample_rate=1_000_000,
        )


# ── Test: channelizer uses input rate ──────────────────────────────────────

def test_channelizer_with_decimation():
    """Channelizer at input rate + decimation = correct frequency shift."""
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV,
        sample_rate=1_000_000,
        input_sample_rate=3_000_000,
        channelizer_offset_hz=500_000.0,
    )
    assert demod._needs_decim
    assert demod._chan_offset == 500_000.0
    assert demod._input_sr == 3_000_000
    assert demod._demod_sr == 1_000_000


# ── Config smoke test: L3 jammer ────────────────────────────────────────────

def test_L3_jammer_config_smoke():
    """
    Verify that with target_jammer_level=3 and rx_source=jammer:
    - plan.sample_rate_hz > phy.sample_rate
    - GFSK2Demodulator would receive input_sample_rate=plan.sample_rate_hz
    - decimation is active
    - channelizer offset is non-zero (L3 is a distinct channel)
    """
    # Simulate what DSPProcessor does with a L3 jammer config
    import json, tempfile, os
    from config_manager import ConfigManager

    # Write a temp config with level=3
    cfg = {
        "team_color": "red",
        "target_jammer_level": 3,
        "phy_mode": "2gfsk",
        "rx_source": "jammer",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        tmp_path = f.name

    try:
        mgr = ConfigManager(tmp_path).load()
        plan = mgr.plan
        phy = mgr.phy_config

        # Level 3 → channelize must be True
        assert plan.jammer_level == 3
        assert plan.channelize

        # SDR sample rate must be >= 3 MSPS
        assert plan.sample_rate_hz >= 3_000_000

        # Demod rate stays at 1 MSPS
        assert phy.sample_rate == 1_000_000

        # Jammer deviation is L3-specific (~105.8 kHz)
        assert 100_000 < phy.jammer_deviation_hz < 111_000

        # Channelizer offset is non-zero (L3 ≠ broadcast)
        assert plan.jammer_offset_hz != 0.0

        # Build GFSK2Demodulator as DSPProcessor would
        rx_dev = phy.jammer_deviation_hz
        chan_offset = plan.jammer_offset_hz  # channelize=True
        demod = GFSK2Demodulator(
            sps=phy.sps,
            bt=phy.bt,
            span=phy.span,
            deviation_hz=rx_dev,
            sample_rate=phy.sample_rate,
            input_sample_rate=float(plan.sample_rate_hz),
            channelizer_offset_hz=chan_offset,
            threshold_mode="zero",
            sub_block_syms=phy.sub_block_syms,
        )

        # Verify decimation is active
        assert demod._needs_decim
        decim = round(plan.sample_rate_hz / phy.sample_rate)
        assert demod._decim == decim
        assert demod._demod_sr == 1_000_000
        assert demod._chan_offset == chan_offset
        assert abs(demod._deviation - rx_dev) < 1.0

    finally:
        os.unlink(tmp_path)


# ── Direct-tune smoke test ─────────────────────────────────────────────────

def test_direct_tune_overrides_input_rate():
    """
    target_jammer_level=3 + rx_freq=434.32:
    GFSK2Demodulator gets input_sample_rate=1MSPS, not 3MSPS.
    """
    import json, tempfile, os
    from config_manager import ConfigManager

    cfg = {
        "team_color": "red",
        "target_jammer_level": 3,
        "phy_mode": "2gfsk",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        tmp = f.name

    try:
        mgr = ConfigManager(tmp).load()
        plan = mgr.plan
        phy = mgr.phy_config

        assert plan.sample_rate_hz >= 3_000_000  # wideband

        # Simulate --rx-freq=434.32: Pluto runs at 1 MSPS
        rx_sr_override = phy.sample_rate  # 1_000_000

        # Simulate DSPProcessor with input_sample_rate_hz override
        rx_dev = phy.jammer_deviation_hz
        demod = GFSK2Demodulator(
            sps=phy.sps,
            bt=phy.bt,
            span=phy.span,
            deviation_hz=rx_dev,
            sample_rate=phy.sample_rate,
            input_sample_rate=rx_sr_override,  # 1 MSPS, NOT 3 MSPS
            channelizer_offset_hz=None,         # direct-tune: no channelizer
            threshold_mode="zero",
        )

        assert not demod._needs_decim
        assert demod._decim == 1
        assert demod._demod_sr == 1_000_000
        assert demod._input_sr == 1_000_000
        assert demod._chan_offset is None

    finally:
        os.unlink(tmp)
