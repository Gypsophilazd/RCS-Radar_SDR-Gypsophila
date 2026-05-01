"""
tests/test_air_packet_deframer.py
==================================
Unit tests for AirPacketDeframer.
"""

import struct
import pytest
from phy.air_packet import AirPacketDeframer, ac_to_bits

# ── helpers ──────────────────────────────────────────────────────────────────

def _bits_of_bytes(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _build_air_packet_bits(ac_int: int, payload: bytes) -> list[int]:
    """
    Build a complete air-packet bit sequence:
      Access Code (64 bits) + Header (0x000F000F, 32 bits) + Payload (15 bytes, 120 bits)
    """
    bits = ac_to_bits(ac_int)
    # Header: two big-endian uint16 both = 15
    header_bytes = struct.pack(">HH", 15, 15)
    bits.extend(_bits_of_bytes(header_bytes))
    bits.extend(_bits_of_bytes(payload))
    return bits


# ── Test: ac_to_bits produces exact known sequence ───────────────────────────

def test_ac_to_bits_info():
    """Verify Access Code info bit sequence matches spec."""
    bits = ac_to_bits(0x2F6F4C74B914492E)
    assert len(bits) == 64
    # First byte 0x2F = 0b00101111
    assert bits[:8] == [0, 0, 1, 0, 1, 1, 1, 1]
    # Last byte 0x2E = 0b00101110
    assert bits[-8:] == [0, 0, 1, 0, 1, 1, 1, 0]
    # Spot-check byte 0x6F = 0b01101111
    assert bits[8:16] == [0, 1, 1, 0, 1, 1, 1, 1]


def test_ac_to_bits_jammer():
    """Verify Access Code jammer bit sequence produces correct byte boundaries."""
    bits = ac_to_bits(0x16E8D377151C712D)
    assert len(bits) == 64
    assert bits[:8] == [0, 0, 0, 1, 0, 1, 1, 0]   # 0x16
    assert bits[-8:] == [0, 0, 1, 0, 1, 1, 0, 1]   # 0x2D


# ── Test: noise before access code ───────────────────────────────────────────

def test_noise_before_ac():
    """Payload extracted when random bits precede the Access Code."""
    payload = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F"
    noise_bits = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]  # 10 bits of noise
    packet_bits = _build_air_packet_bits(0x2F6F4C74B914492E, payload)
    all_bits = noise_bits + packet_bits

    deframer = AirPacketDeframer(mode="info")
    result = deframer.push_bits(all_bits)
    assert len(result) == 1
    assert result[0] == payload


# ── Test: info mode ──────────────────────────────────────────────────────────

def test_info_mode_match():
    """Info mode matches info AC, rejects jammer AC."""
    payload = b"INFO_PACKET_015"
    bits = _build_air_packet_bits(0x2F6F4C74B914492E, payload)
    deframer = AirPacketDeframer(mode="info")
    result = deframer.push_bits(bits)
    assert len(result) == 1
    assert result[0] == payload


def test_info_mode_rejects_jammer_ac():
    """Info mode should NOT match jammer Access Code."""
    payload = b"JAM_PACKET_0015"
    bits = _build_air_packet_bits(0x16E8D377151C712D, payload)
    deframer = AirPacketDeframer(mode="info")
    result = deframer.push_bits(bits)
    assert len(result) == 0


# ── Test: jammer mode ────────────────────────────────────────────────────────

def test_jammer_mode_match():
    """Jammer mode matches jammer AC, rejects info AC."""
    payload = b"JAM_PACKET_0015"
    bits = _build_air_packet_bits(0x16E8D377151C712D, payload)
    deframer = AirPacketDeframer(mode="jammer")
    result = deframer.push_bits(bits)
    assert len(result) == 1
    assert result[0] == payload


def test_jammer_mode_rejects_info_ac():
    """Jammer mode should NOT match info Access Code."""
    payload = b"INFO_PACKET_015"
    bits = _build_air_packet_bits(0x2F6F4C74B914492E, payload)
    deframer = AirPacketDeframer(mode="jammer")
    result = deframer.push_bits(bits)
    assert len(result) == 0


# ── Test: both mode ──────────────────────────────────────────────────────────

def test_both_mode_matches_both():
    """Both mode matches info AND jammer AC."""
    payload_info = b"INFO_PACKET_015"
    payload_jam  = b"JAM_PACKET_0015"
    bits_info = _build_air_packet_bits(0x2F6F4C74B914492E, payload_info)
    bits_jam  = _build_air_packet_bits(0x16E8D377151C712D, payload_jam)
    all_bits = bits_info + bits_jam

    deframer = AirPacketDeframer(mode="both")
    result = deframer.push_bits(all_bits)
    assert len(result) == 2
    assert result[0] == payload_info
    assert result[1] == payload_jam


# ── Test: multiple packets ───────────────────────────────────────────────────

def test_multiple_packets_in_stream():
    """Multiple consecutive air packets are all extracted."""
    payload1 = b"PKT1_1234567890"
    payload2 = b"PKT2_ABCDEFGHIJ"
    payload3 = b"PKT3_klmnopqrst"
    bits = (
        _build_air_packet_bits(0x2F6F4C74B914492E, payload1)
        + _build_air_packet_bits(0x2F6F4C74B914492E, payload2)
        + _build_air_packet_bits(0x2F6F4C74B914492E, payload3)
    )

    deframer = AirPacketDeframer(mode="info")
    # Feed in small chunks to test statefulness
    result = []
    for i in range(0, len(bits), 50):
        result.extend(deframer.push_bits(bits[i:i+50]))

    assert len(result) == 3
    assert result[0] == payload1
    assert result[1] == payload2
    assert result[2] == payload3


# ── Test: bad header rejected ────────────────────────────────────────────────

def test_bad_header_rejected():
    """Header with non-15 values is rejected."""
    ac_bits = ac_to_bits(0x2F6F4C74B914492E)
    # Header with len1=10, len2=10 (not 15,15)
    bad_header_bytes = struct.pack(">HH", 10, 10)
    bad_header_bits = _bits_of_bytes(bad_header_bytes)
    payload = b"123456789012345"
    payload_bits = _bits_of_bytes(payload)

    bits = ac_bits + bad_header_bits + payload_bits
    deframer = AirPacketDeframer(mode="info")
    result = deframer.push_bits(bits)
    assert len(result) == 0


# ── Test: AC exact match ─────────────────────────────────────────────────────

def test_ac_near_miss_not_matched():
    """A single-bit error in the AC should NOT produce a match (exact only)."""
    ac_correct = ac_to_bits(0x2F6F4C74B914492E)
    # Flip last bit of AC
    ac_wrong = ac_correct.copy()
    ac_wrong[-1] ^= 1
    payload = b"123456789012345"
    header_bytes = struct.pack(">HH", 15, 15)
    bits = ac_wrong + _bits_of_bytes(header_bytes) + _bits_of_bytes(payload)
    deframer = AirPacketDeframer(mode="info")
    result = deframer.push_bits(bits)
    assert len(result) == 0


# ── Test: invalid mode ───────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        AirPacketDeframer(mode="invalid")
