"""
tests/test_gfsk2_loopback_0x0A06.py
====================================
End-to-end software loopback: 0x0A06 key "RM2026" through the full
2-GFSK air-link chain — no SDR hardware required.

Chain
─────
  RM frame bytes
  → split into 15-byte air packets (AC + Header + Payload → bits)
  → gfsk2_modulate_bits (IQ samples)
  → GFSK2Demodulator.push_iq (bits)
  → AirPacketDeframer.push_bits (15-byte payloads)
  → PayloadStreamReassembler.push_payload (raw RM frame bytes)
  → packet_decoder.decode_frame (dict)
  → assert key == "RM2026"
"""

import struct
import numpy as np
from packet_decoder import crc8_rm, crc16_rm, decode_frame, SOF
from phy.air_packet import AirPacketDeframer, ac_to_bits
from phy.stream_reassembler import PayloadStreamReassembler
from phy.gfsk2_modem import gfsk2_modulate_bits, GFSK2Demodulator


# ── PHY constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE = 1_000_000
_SPS = 52
_BT = 0.35
_SPAN = 4
_DEVIATION_HZ = 250_000.0
_ACCESS_CODE_INFO = 0x2F6F4C74B914492E
_ACCESS_CODE_JAMMER = 0x16E8D377151C712D


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_rm_frame(cmd_id: int, payload: bytes, seq: int = 1) -> bytes:
    """Build a valid RM frame with CRC8/CRC16."""
    data_bytes = struct.pack("<H", cmd_id) + payload
    data_len = len(data_bytes)
    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, seq])
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + data_bytes
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


def _bytes_to_bits_msb(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _build_air_packet_bits(payload: bytes, ac_int: int = _ACCESS_CODE_INFO) -> list[int]:
    """
    Build a complete air-packet bit sequence:
      Access Code (64 bits) + Header (32 bits, 0x000F000F) + Payload (120 bits).
    """
    assert len(payload) == 15
    bits = ac_to_bits(ac_int)
    header_bytes = struct.pack(">HH", 15, 15)
    bits.extend(_bytes_to_bits_msb(header_bytes))
    bits.extend(_bytes_to_bits_msb(payload))
    return bits


def _chunk_payloads(frame: bytes, payload_size: int = 15) -> list[bytes]:
    """Split a frame into fixed-size payload chunks, padded with zeros."""
    chunks = []
    for i in range(0, len(frame), payload_size):
        chunk = frame[i:i + payload_size]
        if len(chunk) < payload_size:
            chunk = chunk + b"\x00" * (payload_size - len(chunk))
        chunks.append(chunk)
    return chunks


# ── Test: full loopback 0x0A06 ───────────────────────────────────────────────

def test_full_loopback_0x0A06():
    """Complete 2-GFSK loopback: RM frame through air-packet layers and back."""
    # 1. Build RM frame
    frame = _build_rm_frame(0x0A06, b"RM2026")
    assert len(frame) == 15  # SOF(1)+DataLen(2)+Seq(1)+CRC8(1)+CmdID(2)+6Bkey+CRC16(2)

    # 2. Slice into 15-byte air packets
    chunks = _chunk_payloads(frame)
    assert len(chunks) == 1
    assert len(chunks[0]) == 15

    # 3. Build air packet bits
    packet_bits = _build_air_packet_bits(chunks[0])

    # 4. Modulate to IQ
    iq = gfsk2_modulate_bits(
        packet_bits,
        sps=_SPS,
        bt=_BT,
        span=_SPAN,
        deviation_hz=_DEVIATION_HZ,
        sample_rate=_SAMPLE_RATE,
    )

    # 5. Feed through clean channel (no impairments).
    # This validates the end-to-end chain integrity — channel tolerance
    # is tested separately by hardware-in-the-loop and field tests.
    rx_iq = iq.astype(np.complex64)

    # 6. Demodulate
    demod = GFSK2Demodulator(
        sps=_SPS,
        bt=_BT,
        span=_SPAN,
        deviation_hz=_DEVIATION_HZ,
        sample_rate=_SAMPLE_RATE,
        threshold_mode="zero",
    )
    bits = demod.push_iq(rx_iq)

    # 7. Deframe
    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) >= 1, f"Expected >=1 payload, got {len(payloads)}"

    # 8. Reassemble
    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) >= 1, f"Expected >=1 raw frame, got {len(raw_frames)}"

    # 9. Decode
    decoded = decode_frame(raw_frames[0])
    assert decoded is not None, "Frame decode failed (CRC mismatch?)"
    assert decoded["cmd"] == "0x0A06"
    assert decoded["key"] == "RM2026"


# ── Test: loopback with jammer AC ────────────────────────────────────────────

def test_loopback_jammer_ac():
    """Loopback using jammer Access Code."""
    frame = _build_rm_frame(0x0A06, b"JAMKEY")
    chunks = _chunk_payloads(frame)
    packet_bits = _build_air_packet_bits(chunks[0], ac_int=_ACCESS_CODE_JAMMER)

    iq = gfsk2_modulate_bits(
        packet_bits,
        sps=_SPS,
        bt=_BT,
        span=_SPAN,
        deviation_hz=_DEVIATION_HZ,
        sample_rate=_SAMPLE_RATE,
    )

    # Clean channel (no noise/offset for jammer test)
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEVIATION_HZ, sample_rate=_SAMPLE_RATE,
        threshold_mode="zero",
    )
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


# ── Test: multi-packet multi-frame ───────────────────────────────────────────

def test_loopback_multiple_packets():
    """Multiple air packets with different frame fragments reassembled."""
    # 0x0A02 frame: 12-byte payload → frame = 21 bytes (needs 2 air packets)
    hp_payload = struct.pack("<HHHHHH", 100, 200, 300, 400, 500, 600)
    frame = _build_rm_frame(0x0A02, hp_payload)
    assert len(frame) == 21

    # Split into 2 payloads: 15 + 6(padded to 15)
    chunks = _chunk_payloads(frame)
    assert len(chunks) == 2

    # Build air packets for both chunks
    all_iq_parts = []
    for ch in chunks:
        packet_bits = _build_air_packet_bits(ch)
        iq = gfsk2_modulate_bits(
            packet_bits,
            sps=_SPS, bt=_BT, span=_SPAN,
            deviation_hz=_DEVIATION_HZ, sample_rate=_SAMPLE_RATE,
        )
        all_iq_parts.append(iq)

    # Concatenate IQ
    full_iq = np.concatenate(all_iq_parts)

    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEVIATION_HZ, sample_rate=_SAMPLE_RATE,
        threshold_mode="zero",
    )
    bits = demod.push_iq(full_iq)

    deframer = AirPacketDeframer(mode="info")
    payloads = deframer.push_bits(bits)
    assert len(payloads) == 2

    reassembler = PayloadStreamReassembler()
    raw_frames = []
    for p in payloads:
        raw_frames.extend(reassembler.push_payload(p))
    assert len(raw_frames) == 1
    assert raw_frames[0] == frame


# ── Test: 2-GFSK path uses binary slicer only ────────────────────────────────

def test_gfsk2_binary_slicer_one_bit_per_symbol():
    """Verify GFSK2 demodulator outputs exactly 1 bit per symbol."""
    # Generate some test bits
    test_bits = [1, 0, 1, 1, 0, 0, 1, 0] * 50  # 400 bits

    iq = gfsk2_modulate_bits(
        test_bits,
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEVIATION_HZ, sample_rate=_SAMPLE_RATE,
    )

    # Insert a spy to count symbols processed
    demod = GFSK2Demodulator(
        sps=_SPS, bt=_BT, span=_SPAN,
        deviation_hz=_DEVIATION_HZ, sample_rate=_SAMPLE_RATE,
        threshold_mode="zero",
    )

    # Count symbols from clock recovery
    class _SpyClock:
        def __init__(self, real_clock):
            self._real = real_clock
            self.symbol_count = 0

        def process(self, samples):
            syms = self._real.process(samples)
            self.symbol_count += len(syms)
            return syms

    spy = _SpyClock(demod._clock)
    demod._clock = spy

    bits = demod.push_iq(iq)
    # symbol count should equal bits count
    assert spy.symbol_count == len(bits), (
        f"Symbol count {spy.symbol_count} != bit count {len(bits)}"
    )
    # Bits are only 0 or 1 (binary)
    assert all(b in (0, 1) for b in bits)
