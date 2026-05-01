"""
phy/air_packet.py
=================
AirPacketDeframer — 2-GFSK air-link packet deframer.

Searches demodulated bit stream for 64-bit Access Code, validates the
32-bit Header (two big-endian uint16 payload-length fields, both must
equal 15), and extracts the 15-byte payload.

Access Code bit order (explicit and testable):
  1. Integer literal: 0x2F6F4C74B914492E
  2. Convert to 8 bytes big-endian
  3. Per byte: MSB-first bit serialisation
  → 64-bit reference pattern stored as a Python int for shift-register matching.

Usage::

    deframer = AirPacketDeframer(mode="both")
    payloads = deframer.push_bits(bits)   # list of 15-byte payloads
"""

from __future__ import annotations

import struct
from typing import List

# ── Access Codes (64-bit integer literals) ────────────────────────────────────
_AC_INFO   = 0x2F6F4C74B914492E
_AC_JAMMER = 0x16E8D377151C712D

# Header expected values
_HEADER_LEN1 = 15
_HEADER_LEN2 = 15
# Header as two big-endian uint16: 0x000F 0x000F → 32 bits
_HEADER_BITS = 32
_HEADER_VALUE = 0x000F000F  # both uint16 == 15

_PAYLOAD_BYTES = 15
_PAYLOAD_BITS  = 120
_AC_BITS       = 64

# Bit mask for 64-bit shift register
_AC_MASK64 = (1 << 64) - 1
_HEADER_MASK32 = (1 << 32) - 1


def ac_to_bits(ac_int: int) -> list[int]:
    """
    Convert a 64-bit Access Code integer to its on-air bit sequence.

    Algorithm
    ─────────
    1. Pack as 8 bytes big-endian
    2. Serialise each byte MSB-first

    Returns a list of 64 bits (0/1 ints).

    >>> ac_to_bits(0x2F6F4C74B914492E)[:8]
    [0, 0, 1, 0, 1, 1, 1, 1]   # 0x2F
    """
    ac_bytes = struct.pack(">Q", ac_int)
    bits: list[int] = []
    for b in ac_bytes:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _bits_to_int(bits: list[int]) -> int:
    """Pack bit list (MSB-first) into integer — used for header parsing."""
    val = 0
    for b in bits:
        val = (val << 1) | (b & 1)
    return val


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Pack bit list (MSB-first per byte, length must be multiple of 8)."""
    assert len(bits) % 8 == 0, f"Bit count {len(bits)} not byte-aligned"
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)


class AirPacketDeframer:
    """
    Sliding-window Access Code deframer for 2-GFSK air packets.

    Parameters
    ----------
    mode : str
        "info"   — match only info Access Code  (0x2F6F4C74B914492E)
        "jammer" — match only jammer Access Code (0x16E8D377151C712D)
        "both"   — match either Access Code

    State
    ─────
    The deframer maintains a 64-bit shift register and a small internal
    bit buffer for accumulating header + payload bits after a match.
    """

    def __init__(self, mode: str = "both"):
        if mode not in ("info", "jammer", "both"):
            raise ValueError(f"mode must be 'info', 'jammer', or 'both', got {mode!r}")
        self._mode = mode

        # Pre-compute the expected AC bit-patterns as 64-bit integers
        self._ac_info_bits   = _bits_to_int(ac_to_bits(_AC_INFO))
        self._ac_jammer_bits = _bits_to_int(ac_to_bits(_AC_JAMMER))

        # 64-bit sliding shift register for AC hunt
        self._sreg: int = 0
        self._sreg_count: int = 0   # number of bits pushed into sreg

        # Post-AC state
        # Possible states: "hunt", "header", "payload"
        self._state: str = "hunt"
        self._bit_buf: list[int] = []   # accumulates header/payload bits
        self._need_bits: int = 0        # bits remaining in current state

    # ── public ──────────────────────────────────────────────────────────────

    def push_bits(self, bits: list[int]) -> list[bytes]:
        """
        Feed demodulated bits and return any completed 15-byte payloads.

        Parameters
        ----------
        bits : list of int (0/1)

        Returns
        -------
        list[bytes] — zero or more 15-byte payload chunks
        """
        out: list[bytes] = []
        for b in bits:
            bit = b & 1
            self._sreg = ((self._sreg << 1) | bit) & _AC_MASK64
            self._sreg_count += 1

            if self._state == "hunt":
                if self._sreg_count >= _AC_BITS and self._match_ac():
                    self._state = "header"
                    self._bit_buf = []
                    self._need_bits = _HEADER_BITS

            elif self._state == "header":
                self._bit_buf.append(bit)
                self._need_bits -= 1
                if self._need_bits == 0:
                    header_int = _bits_to_int(self._bit_buf)
                    len1 = (header_int >> 16) & 0xFFFF
                    len2 = header_int & 0xFFFF
                    if len1 == _HEADER_LEN1 and len2 == _HEADER_LEN2:
                        self._state = "payload"
                        self._bit_buf = []
                        self._need_bits = _PAYLOAD_BITS
                    else:
                        self._state = "hunt"

            elif self._state == "payload":
                self._bit_buf.append(bit)
                self._need_bits -= 1
                if self._need_bits == 0:
                    payload = _bits_to_bytes(self._bit_buf)
                    assert len(payload) == _PAYLOAD_BYTES
                    out.append(payload)
                    self._state = "hunt"

        return out

    # ── private ─────────────────────────────────────────────────────────────

    def _match_ac(self) -> bool:
        """Check if the current 64-bit shift register matches any enabled AC."""
        if self._mode in ("info", "both"):
            if self._sreg == self._ac_info_bits:
                return True
        if self._mode in ("jammer", "both"):
            if self._sreg == self._ac_jammer_bits:
                return True
        return False
