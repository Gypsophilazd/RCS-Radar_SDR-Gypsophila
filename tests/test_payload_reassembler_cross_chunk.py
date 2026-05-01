"""
tests/test_payload_reassembler_cross_chunk.py
==============================================
Unit tests for PayloadStreamReassembler — cross-chunk RM frame reassembly.
"""

import struct
import pytest
from packet_decoder import crc8_rm, crc16_rm, decode_frame, SOF
from phy.stream_reassembler import PayloadStreamReassembler


def _build_rm_frame(cmd_id: int, payload: bytes, seq: int = 1) -> bytes:
    """Build a valid RM frame with correct CRC8 and CRC16."""
    data_bytes = struct.pack("<H", cmd_id) + payload
    data_len = len(data_bytes)

    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, seq])
    crc8_val = crc8_rm(header)

    crc16_input = header + bytes([crc8_val]) + data_bytes
    crc16_val = crc16_rm(crc16_input)

    frame = crc16_input + struct.pack("<H", crc16_val)
    return frame


def _chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    """Split bytes into fixed-size chunks, last chunk may be smaller."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


# ── Single frame in one payload ──────────────────────────────────────────────

def test_single_frame_one_payload():
    """A complete RM frame fits in a single 15-byte payload."""
    frame = _build_rm_frame(0x0A06, b"RM2026")  # 6-byte key → 15-byte frame
    assert len(frame) == 15
    # Frame is exactly 15 bytes — fits perfectly in one air packet payload
    payload = frame

    reassembler = PayloadStreamReassembler()
    results = reassembler.push_payload(payload)
    assert len(results) == 1
    assert results[0] == frame


# ── Frame split across two payloads ──────────────────────────────────────────

def test_frame_split_two_payloads():
    """RM frame split across two 15-byte payload chunks."""
    # Use 0x0A02 frame (12-byte payload): data_len=14, frame=21 bytes
    hp_payload = struct.pack("<HHHHHH", 100, 200, 300, 400, 500, 600)
    frame = _build_rm_frame(0x0A02, hp_payload)
    assert len(frame) == 21

    # Chunk1: first 15 bytes of frame (exact payload size, no padding)
    chunk1 = frame[:15]
    # Chunk2: last 6 bytes of frame, padded to 15
    chunk2 = frame[15:] + b"\x00" * 9

    reassembler = PayloadStreamReassembler()
    r1 = reassembler.push_payload(chunk1)
    assert len(r1) == 0  # not enough bytes yet (15 < 21)

    r2 = reassembler.push_payload(chunk2)
    assert len(r2) == 1
    assert r2[0] == frame


# ── Frame split across three payloads ────────────────────────────────────────

def test_frame_split_three_payloads():
    """RM frame split across three 15-byte payload chunks."""
    # Use a larger frame (0x0A05 has 36-byte payload, total = 5+36+2 = 43 bytes)
    buff_payload = bytes([0] * 36)
    frame = _build_rm_frame(0x0A05, buff_payload)
    # data_len = 2 + 36 = 38, frame = 38 + 7 = 45
    assert len(frame) == 45

    # Split into 3 chunks of 15 bytes (frame bytes: 15+15+15, exactly 3 chunks)
    chunk1 = frame[:15]
    chunk2 = frame[15:30]
    chunk3 = frame[30:]  # exactly 15 bytes, no padding needed

    reassembler = PayloadStreamReassembler()
    assert reassembler.push_payload(chunk1) == []
    assert reassembler.push_payload(chunk2) == []
    results = reassembler.push_payload(chunk3)
    assert len(results) == 1
    assert results[0] == frame


# ── Multiple frames in stream ────────────────────────────────────────────────

def test_two_frames_in_stream():
    """Two consecutive RM frames recovered from payload stream."""
    frame1 = _build_rm_frame(0x0A06, b"KEY001")  # 6-byte key → 15-byte frame
    frame2 = _build_rm_frame(0x0A06, b"KEY002")
    assert len(frame1) == 15 and len(frame2) == 15

    # Pack both frames into sequential 15-byte payloads
    reassembler = PayloadStreamReassembler()
    r1 = reassembler.push_payload(frame1)  # exact 15-byte frame
    assert len(r1) == 1
    assert r1[0] == frame1

    r2 = reassembler.push_payload(frame2)
    assert len(r2) == 1
    assert r2[0] == frame2


# ── Noise before SOF ─────────────────────────────────────────────────────────

def test_noise_before_sof():
    """Random bytes before SOF are discarded."""
    frame = _build_rm_frame(0x0A06, b"RM2026")  # 6-byte key → 15-byte frame
    assert len(frame) == 15
    # 3 bytes of noise + 12 bytes of frame = 15 byte payload
    payload = bytes([0xFF, 0x00, 0xAA]) + frame[:12]

    reassembler = PayloadStreamReassembler()
    r1 = reassembler.push_payload(payload)
    # Only 12 bytes of the frame delivered; need 3 more
    assert r1 == []

    # Second payload: remainder of frame
    payload2 = frame[12:] + b"\x00" * 12
    r2 = reassembler.push_payload(payload2)
    assert len(r2) == 1
    assert r2[0] == frame


# ── CRC failure ──────────────────────────────────────────────────────────────

def test_crc8_failure_discarded():
    """Frame with bad CRC8 header is discarded."""
    frame = _build_rm_frame(0x0A06, b"RM2026")
    # Corrupt CRC8 byte (index 4)
    corrupted = bytearray(frame)
    corrupted[4] ^= 0xFF
    payload = bytes(corrupted)  # 15 bytes fits exactly

    reassembler = PayloadStreamReassembler()
    results = reassembler.push_payload(payload)
    assert results == []


def test_crc16_failure_discarded():
    """Frame with bad CRC16 tail is discarded."""
    frame = _build_rm_frame(0x0A06, b"RM2026")
    # Corrupt CRC16 (last 2 bytes)
    corrupted = bytearray(frame)
    corrupted[-1] ^= 0xFF
    payload = bytes(corrupted)

    reassembler = PayloadStreamReassembler()
    results = reassembler.push_payload(payload)
    assert results == []


# ── Buffer overflow protection ───────────────────────────────────────────────

def test_max_buffer_truncation():
    """Stream buffer does not grow beyond max_buffer."""
    reassembler = PayloadStreamReassembler(max_buffer=30)
    # Feed 5 payloads of noise (no SOF found → entire buffer is noise)
    for _ in range(5):
        reassembler.push_payload(b"\xFF" * 15)
    # Buffer should be ≤ max_buffer
    assert len(reassembler._stream) <= 30


# ── Frame decoded correctly ──────────────────────────────────────────────────

def test_frame_decoded_correctly():
    """Raw bytes from reassembler decode to correct dict."""
    frame = _build_rm_frame(0x0A06, b"RM2026")

    reassembler = PayloadStreamReassembler()
    results = reassembler.push_payload(frame)
    assert len(results) == 1

    decoded = decode_frame(results[0])
    assert decoded is not None
    assert decoded["cmd"] == "0x0A06"
    assert decoded["key"] == "RM2026"
    assert decoded["seq"] == 1
