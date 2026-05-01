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
    mgr = load_config()
    plan = mgr.plan
    assert plan.jammer_level == 0
    assert plan.sample_rate_hz == 1_000_000
    assert not plan.channelize


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
    # The key invariant: when channelizer_offset_hz is given AND rates differ,
    # the channelizer shifts at the INPUT rate, then decimation reduces to
    # the DEMOD rate where FM discriminator operates.
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEV,
        sample_rate=1_000_000,
        input_sample_rate=3_000_000,
        channelizer_offset_hz=500_000.0,  # shift by 500 kHz
    )
    assert demod._needs_decim
    assert demod._chan_offset == 500_000.0
    # Channelizer operates at input_sr (3e6), FM discrim at demod_sr (1e6)
    assert demod._input_sr == 3_000_000
    assert demod._demod_sr == 1_000_000
