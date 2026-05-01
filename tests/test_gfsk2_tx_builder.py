"""
tests/test_gfsk2_tx_builder.py
===============================
Unit tests for phy/gfsk2_tx_builder — no SDR hardware required.
"""

import struct
import numpy as np
from packet_decoder import crc8_rm, crc16_rm, SOF, decode_frame
from phy.gfsk2_tx_builder import (
    build_gfsk2_tx_iq,
    build_air_packet_bits_from_payload,
    AC_INFO,
    AC_JAMMER,
)
from phy.air_packet import ac_to_bits, AirPacketDeframer
from phy.gfsk2_modem import GFSK2Demodulator
from phy.stream_reassembler import PayloadStreamReassembler


def _build_rm_frame(cmd_id: int, payload: bytes, seq: int = 1) -> bytes:
    data_bytes = struct.pack("<H", cmd_id) + payload
    data_len = len(data_bytes)
    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, seq])
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + data_bytes
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


# ── Test: RM frame → 15B air packet bits ────────────────────────────────────

def test_air_packet_bits_format():
    """Single 15-byte payload produces exactly 216 bits (64+32+120)."""
    payload = b"0123456789ABCDE"
    assert len(payload) == 15
    bits = build_air_packet_bits_from_payload(payload, AC_INFO)
    assert len(bits) == 216

    # First 64 bits must match AC bit pattern
    expected_ac = ac_to_bits(AC_INFO)
    assert bits[:64] == expected_ac

    # Header 0x000F000F = [00, 0F, 00, 0F] as big-endian uint16 pair
    # MSB-first per byte: 0x00→[0]*8, 0x0F→[0,0,0,0,1,1,1,1]
    assert bits[64:72] == [0, 0, 0, 0, 0, 0, 0, 0]   # 0x00
    assert bits[72:80] == [0, 0, 0, 0, 1, 1, 1, 1]   # 0x0F


# ── Test: builder loopback ──────────────────────────────────────────────────

def test_builder_loopback_info():
    """2-GFSK TX builder → demod → deframer → reassemble → decode (info AC)."""
    frame = _build_rm_frame(0x0A06, b"RM2026")
    iq = build_gfsk2_tx_iq(frame, mode="info", repeats=1)
    # Remove DAC scaling for demod
    iq = iq.astype(np.complex64) / (2**14 * 0.5)

    demod = GFSK2Demodulator(sps=52, bt=0.35, span=4, deviation_hz=250000.0, sample_rate=1e6)
    bits = demod.push_iq(iq)
    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) >= 1

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) >= 1

    decoded = decode_frame(raw_frames[0])
    assert decoded is not None
    assert decoded["key"] == "RM2026"


def test_builder_loopback_jammer():
    """2-GFSK TX builder loopback with jammer AC."""
    frame = _build_rm_frame(0x0A06, b"JAMKEY")
    iq = build_gfsk2_tx_iq(frame, mode="jammer", repeats=1)
    iq = iq.astype(np.complex64) / (2**14 * 0.5)

    demod = GFSK2Demodulator(sps=52, bt=0.35, span=4, deviation_hz=250000.0, sample_rate=1e6)
    bits = demod.push_iq(iq)
    deframer = AirPacketDeframer(mode="jammer")
    payloads = deframer.push_bits(bits)
    assert len(payloads) >= 1

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) >= 1

    decoded = decode_frame(raw_frames[0])
    assert decoded is not None
    assert decoded["key"] == "JAMKEY"


# ── Test: multi-packet frame ─────────────────────────────────────────────────

def test_builder_multi_packet():
    """Frame larger than 15 bytes uses multiple air packets."""
    # 0x0A02: 12-byte payload → frame = 21 bytes → 2 air packets
    hp = struct.pack("<HHHHHH", 100, 200, 300, 400, 500, 600)
    frame = _build_rm_frame(0x0A02, hp)
    assert len(frame) == 21

    iq = build_gfsk2_tx_iq(frame, mode="info", repeats=1)
    iq = iq.astype(np.complex64) / (2**14 * 0.5)

    demod = GFSK2Demodulator(sps=52, bt=0.35, span=4, deviation_hz=250000.0, sample_rate=1e6)
    bits = demod.push_iq(iq)
    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) == 2

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) == 1
    assert raw_frames[0] == frame


# ── Test: TX/RX parameter consistency ────────────────────────────────────────

def test_deviation_from_config():
    """Builder deviation_hz can be sourced from ConfigManager.phy_config."""
    from config_manager import load_config
    mgr = load_config()
    phy = mgr.phy_config

    # For 2gfsk mode, deviation must be computed from sensitivity
    assert phy.mode == "2gfsk"
    import math
    expected = phy.sensitivity * phy.sample_rate / (2.0 * math.pi)
    assert abs(phy.deviation_hz - expected) < 500.0  # within 500 Hz


def test_tx_builder_no_legacy_imports():
    """2-GFSK builder must NOT import legacy 4-FSK code."""
    import phy.gfsk2_tx_builder as mod
    source = mod.__dict__
    # Check: no reference to 4-FSK symbols, RRC, fsk_digital_twin
    assert "_bits_to_symbols" not in source
    assert "_make_rrc" not in source
    assert "fsk_digital_twin" not in str(getattr(mod, "__doc__", ""))
