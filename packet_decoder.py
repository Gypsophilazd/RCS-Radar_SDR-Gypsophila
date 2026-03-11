"""
packet_decoder.py
=================
Module 4b — Protocol Frame Parser
RCS-Radar-SDR  RM2026 Arena System

Parses raw frame bytes recovered by the DSP layer into typed Python dicts.
Supports all six RM2026 wireless-link CmdIDs (0x0A01 – 0x0A06).

Frame layout (little-endian)
─────────────────────────────
  Offset  Size  Field
  ──────  ────  ──────────────────────────────────────────
    0       1   SOF  = 0xA5
    1       2   DataLen (LE) — byte count of  CmdID + Payload
    3       1   Seq
    4       1   CRC8  (polynomial 0x31, init 0xFF) over bytes[0:4]
    5       2   CmdID (LE)
    7     var   Payload  (DataLen − 2 bytes)
   end      2   CRC16 (polynomial 0x1021, init 0xFFFF, LE) over bytes[0:-2]

Payload sizes (CMDS_LEN, exclusive of CmdID overhead)
───────────────────────────────────────────────────────
  0x0A01  24 B  Enemy positions    — 5 × (uint16 x_cm, uint16 y_cm) + 4 B flags
  0x0A02  12 B  Enemy HP           — 6 × uint16
  0x0A03  10 B  Enemy ammo         — 5 × uint16
  0x0A04   8 B  Macro status       — uint16 gold, uint16 outpost_hp, uint32 zone_bits
  0x0A05  36 B  Enemy buffs        — 6 units × 6 × uint8
  0x0A06   6 B  Jammer key         — char[6] ASCII, null-padded

  NOTE: The exact field layout for 0x0A01–0x0A05 follows the best available
  interpretation of the RM2026 referee spec.  If Dji releases an updated
  spec with different byte offsets, edit the corresponding parse_0x0AXX()
  function only — all CRC logic and the dispatcher are unaffected.

Usage::

    from packet_decoder import decode_frame

    result = decode_frame(raw_bytes)
    if result:
        print(result)   # {'cmd': '0x0A06', 'key': 'RM2026', 'seq': 1}
"""

from __future__ import annotations

import struct
from typing import Optional

# ─── CRC implementations (must match TX side) ──────────────────────────────────

def _crc8_rm(data: bytes) -> int:
    """CRC-8 (poly=0x31, init=0xFF, MSB-first) — matches RoboMaster referee spec."""
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _crc16_rm(data: bytes) -> int:
    """CRC-16/CCITT-FALSE  (poly=0x1021, init=0xFFFF, refin=False, refout=False)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc


# ─── Protocol constants ─────────────────────────────────────────────────────────

SOF          = 0xA5
MIN_FRAME    = 9           # SOF(1)+DataLen(2)+Seq(1)+CRC8(1)+CmdID(2)+CRC16(2)

# CmdID → expected payload length in bytes (excludes the 2-byte CmdID itself)
CMDS_LEN: dict[int, int] = {
    0x0A01: 24,   # Enemy positions
    0x0A02: 12,   # Enemy HP
    0x0A03: 10,   # Enemy ammo
    0x0A04:  8,   # Macro status
    0x0A05: 36,   # Enemy buffs
    0x0A06:  6,   # Jammer key
}

# Robot slot indices used across 0x0A01 – 0x0A05
_SLOTS = ("hero", "engineer", "infantry_3", "infantry_4", "infantry_5", "sentry")

# ─── Individual parsers ─────────────────────────────────────────────────────────

def parse_0x0A01(payload: bytes) -> dict:
    """
    Enemy Positions (24 bytes)
    ──────────────────────────
    5 ground units × (uint16 x_cm, uint16 y_cm) = 20 bytes
    uint16  valid_mask   — bit N=1 means slot N coordinates are fresh
    uint16  _reserved

    Units: centimetres.  Convert to metres by dividing by 100.
    Slots  0..4 → hero, engineer, infantry_3, infantry_4, infantry_5
    (Sentry does not broadcast ground coordinates.)
    """
    if len(payload) < 24:
        raise ValueError(f"0x0A01 payload too short: {len(payload)} < 24")
    positions = {}
    for i, name in enumerate(_SLOTS[:5]):
        x_cm, y_cm = struct.unpack_from("<HH", payload, i * 4)
        positions[name] = (round(x_cm / 100, 2), round(y_cm / 100, 2))
    valid_mask, = struct.unpack_from("<H", payload, 20)
    return {
        "cmd":        "0x0A01",
        "positions":  positions,           # {slot: (x_m, y_m)}
        "valid_mask": f"0x{valid_mask:04X}",
    }


def parse_0x0A02(payload: bytes) -> dict:
    """
    Enemy HP (12 bytes)
    ────────────────────
    6 units × uint16 current_hp
    Slots: hero, engineer, infantry_3, infantry_4, infantry_5, sentry
    """
    if len(payload) < 12:
        raise ValueError(f"0x0A02 payload too short: {len(payload)} < 12")
    hp = {}
    for i, name in enumerate(_SLOTS):
        val, = struct.unpack_from("<H", payload, i * 2)
        hp[name] = val
    return {"cmd": "0x0A02", "hp": hp}


def parse_0x0A03(payload: bytes) -> dict:
    """
    Enemy Ammo (10 bytes)
    ──────────────────────
    5 units × uint16 remaining_bullets (shoot-allowance counter)
    Slots: hero, infantry_3, infantry_4, infantry_5, sentry
    (Engineer has no shooter.)
    """
    if len(payload) < 10:
        raise ValueError(f"0x0A03 payload too short: {len(payload)} < 10")
    shooters = ("hero", "infantry_3", "infantry_4", "infantry_5", "sentry")
    ammo = {}
    for i, name in enumerate(shooters):
        val, = struct.unpack_from("<H", payload, i * 2)
        ammo[name] = val
    return {"cmd": "0x0A03", "ammo": ammo}


def parse_0x0A04(payload: bytes) -> dict:
    """
    Macro Status (8 bytes)
    ──────────────────────
    uint16  gold_coins     — team coin balance
    uint16  outpost_hp     — opponent outpost remaining HP
    uint32  zone_bitmap    — bitmask of contested/occupied gain zones
                             Bit layout (LSB first):
                               0 = Supply zone    1 = Elevated platform
                               2 = Flying ramp    3 = Central highlands
                               4..31 = reserved / future zones
    """
    if len(payload) < 8:
        raise ValueError(f"0x0A04 payload too short: {len(payload)} < 8")
    gold, outpost_hp, zone_bits = struct.unpack_from("<HHI", payload, 0)
    zones = {
        "supply_zone":       bool(zone_bits & (1 << 0)),
        "elevated_platform": bool(zone_bits & (1 << 1)),
        "flying_ramp":       bool(zone_bits & (1 << 2)),
        "central_highlands": bool(zone_bits & (1 << 3)),
        "_raw":              f"0x{zone_bits:08X}",
    }
    return {
        "cmd":        "0x0A04",
        "gold":       gold,
        "outpost_hp": outpost_hp,
        "zones":      zones,
    }


def parse_0x0A05(payload: bytes) -> dict:
    """
    Enemy Buffs (36 bytes)
    ──────────────────────
    6 units × 6 × uint8 buff-level (0 = inactive, 1/2/3 = tier)

    Buff slot order per unit:
      0 hp_regen   1 cooling     2 defense
      3 vulnerable 4 attack      5 posture  (sentry only: 0=normal,1=patrol,2=guard)
    """
    if len(payload) < 36:
        raise ValueError(f"0x0A05 payload too short: {len(payload)} < 36")
    _BUFF_FIELDS = ("hp_regen", "cooling", "defense", "vulnerable", "attack", "posture")
    buffs = {}
    for i, unit in enumerate(_SLOTS):
        base = i * 6
        unit_buffs = {
            field: payload[base + j]
            for j, field in enumerate(_BUFF_FIELDS)
        }
        buffs[unit] = unit_buffs
    return {"cmd": "0x0A05", "buffs": buffs}


def parse_0x0A06(payload: bytes) -> dict:
    """
    Jammer Key (6 bytes)
    ─────────────────────
    char[6]  key  — ASCII, zero-padded to 6 bytes
    Use this key to configure our own jammer to the matching frequency.
    """
    if len(payload) < 6:
        raise ValueError(f"0x0A06 payload too short: {len(payload)} < 6")
    key = payload[:6].rstrip(b"\x00").decode("ascii", errors="replace")
    return {"cmd": "0x0A06", "key": key}


# ─── Dispatcher ─────────────────────────────────────────────────────────────────

_PARSERS: dict[int, callable] = {
    0x0A01: parse_0x0A01,
    0x0A02: parse_0x0A02,
    0x0A03: parse_0x0A03,
    0x0A04: parse_0x0A04,
    0x0A05: parse_0x0A05,
    0x0A06: parse_0x0A06,
}


def verify_frame(raw: bytes) -> bool:
    """Return True when both CRC8 (header) and CRC16 (full frame) pass."""
    if len(raw) < MIN_FRAME:
        return False
    if raw[0] != SOF:
        return False
    if _crc8_rm(raw[0:4]) != raw[4]:
        return False
    crc16_rx, = struct.unpack_from("<H", raw, len(raw) - 2)
    if _crc16_rm(raw[:-2]) != crc16_rx:
        return False
    return True


def decode_frame(raw: bytes) -> Optional[dict]:
    """
    Full frame decoder.

    Parameters
    ----------
    raw : complete frame bytes (SOF … CRC16 inclusive)

    Returns
    -------
    dict  with at minimum {'cmd': '0x0AXX', 'seq': N, …}
    None  if CRC fails or the frame is malformed
    """
    if not verify_frame(raw):
        return None

    seq    = raw[3]
    cmd_id, = struct.unpack_from("<H", raw, 5)
    payload = raw[7: len(raw) - 2]          # everything between CmdID and CRC16

    # Validate payload length if we know the expected size
    expected_len = CMDS_LEN.get(cmd_id)
    if expected_len is not None and len(payload) != expected_len:
        # Corrupted or spec-mismatch — reject rather than parse garbage
        return None

    parser = _PARSERS.get(cmd_id)
    if parser is None:
        # Unknown CmdID: return raw hex for debugging / future extension
        return {"cmd": f"0x{cmd_id:04X}", "seq": seq, "raw_payload": payload.hex()}

    result = parser(payload)
    result["seq"] = seq
    return result


# ─── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a valid 0x0A06 frame by hand and round-trip it.
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from fsk_digital_twin import build_protocol_frame  # type: ignore

    frame = build_protocol_frame("RM2026")
    result = decode_frame(bytes(frame))
    print("Round-trip test:")
    print(f"  Raw  : {bytes(frame).hex()}")
    print(f"  Parse: {result}")
    assert result is not None and result["key"] == "RM2026", "Round-trip FAILED"
    print("  [PASS]")
