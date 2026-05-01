#!/usr/bin/env python3
"""
tx_gfsk2_test.py
================
Standalone 2-GFSK Transmitter — RM2026 Air-Link Test Signal Generator.

Builds RM protocol frames, slices them into 15-byte air-packet payloads,
wraps them with Access Code + Header, modulates via 2-GFSK, and transmits
continuously via ADALM-PLUTO in cyclic-buffer mode.

Usage
─────
  # Info wave (listen with AirPacketDeframer mode="info")
  python3 tx_gfsk2_test.py --test-tx-enable --key RM2026

  # Jammer wave (listen with AirPacketDeframer mode="jammer")
  python3 tx_gfsk2_test.py --test-tx-enable --key RM2026 --mode jammer

  # Custom frequency override
  python3 tx_gfsk2_test.py --test-tx-enable --freq 433.92

  # Low power for benchtop (TX attenuation in dB)
  python3 tx_gfsk2_test.py --test-tx-enable --attenuation 30

Cross-platform
──────────────
  Copy phy/ + packet_decoder.py + tx_gfsk2_test.py to the TX machine.
  Install: pip install numpy scipy pyadi-iio
  Run with the same parameters.  RF signals are OS-independent.

Hardware requirements
─────────────────────
  - ADALM-PLUTO (AD9363) connected via USB/RNDIS
  - 433 MHz antenna or direct SMA cable (for benchtop loopback)
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from repo root even when script is run from another directory
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from packet_decoder import crc8_rm, crc16_rm, SOF
from phy.gfsk2_modem import gfsk2_modulate_bits
from phy.air_packet import ac_to_bits

# ── PHY constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE    = 1_000_000
_SPS            = 52
_BT             = 0.35
_SPAN           = 4
_DEVIATION_HZ   = 250_000.0

_ACCESS_CODE_INFO   = 0x2F6F4C74B914492E
_ACCESS_CODE_JAMMER = 0x16E8D377151C712D

# ── Frequencies (Hz) ─────────────────────────────────────────────────────────
_BROADCAST_FREQ: dict[str, float] = {
    "blue": 433_920_000.0,
    "red":  433_200_000.0,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-GFSK Air-Link Test Transmitter — RM2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tx_gfsk2_test.py --test-tx-enable
  python3 tx_gfsk2_test.py --test-tx-enable --key MYKEY --mode jammer
  python3 tx_gfsk2_test.py --test-tx-enable --freq 433.92 --attenuation 20
        """,
    )
    p.add_argument("--test-tx-enable", action="store_true",
                   help="REQUIRED: enable TX (safety interlock)")
    p.add_argument("--key", default="RM2026", metavar="KEY",
                   help="6-char ASCII jammer key (default: RM2026)")
    p.add_argument("--freq", type=float, default=None, metavar="MHz",
                   help="TX frequency in MHz (default: auto from config team_color)")
    p.add_argument("--mode", default="info", choices=["info", "jammer"],
                   help="Access Code type: info or jammer (default: info)")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Path to config.json (default: ./config.json)")
    p.add_argument("--uri", default=None, metavar="URI",
                   help="Pluto URI (default: from config.json pluto_uri)")
    p.add_argument("--attenuation", type=float, default=None, metavar="dB",
                   help="TX attenuation 0..89 dB (default: from config.json)")
    p.add_argument("--repeats", type=int, default=4, metavar="N",
                   help="Frame repetitions in cyclic buffer (default: 4)")
    return p.parse_args()


def _load_config(config_path: str | None) -> dict:
    """Load config.json, return dict with fallback defaults."""
    import json
    path = Path(config_path) if config_path else _REPO / "config.json"
    raw = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    return raw


def _build_rm_frame(key: str) -> bytes:
    """Build a valid RM frame (CmdID=0x0A06) with CRC8/CRC16."""
    cmd_id = 0x0A06
    key_bytes = key.encode("ascii")[:6].ljust(6, b"\x00")
    payload = struct.pack("<H", cmd_id) + key_bytes
    data_len = len(payload)

    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, 1])  # seq=1
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + payload
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


def _bytes_to_bits_msb(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits: list[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _build_air_packet_bits(payload: bytes, ac_int: int) -> list[int]:
    """Build air-packet bits: AC(64) + Header(32) + Payload(120)."""
    assert len(payload) == 15, f"Payload must be 15 bytes, got {len(payload)}"
    bits = ac_to_bits(ac_int)
    header_bytes = struct.pack(">HH", 15, 15)   # two big-endian uint16
    bits.extend(_bytes_to_bits_msb(header_bytes))
    bits.extend(_bytes_to_bits_msb(payload))
    return bits


def main() -> None:
    args = _parse_args()

    # ── Safety interlock ─────────────────────────────────────────────────────
    if not args.test_tx_enable:
        raise RuntimeError(
            "TX is disabled by default. "
            "Use --test-tx-enable only in legal test conditions."
        )

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = _load_config(args.config)
    our_color = str(cfg.get("team_color", "red")).lower().strip()

    # Determine frequency
    if args.freq is not None:
        tx_freq_hz = int(args.freq * 1e6)
    else:
        tx_freq_hz = int(_BROADCAST_FREQ.get(our_color, 433_200_000))

    # Determine URI
    tx_uri = args.uri or str(cfg.get("pluto_uri", "ip:192.168.2.1"))

    # Determine attenuation
    tx_atten = args.attenuation if args.attenuation is not None else int(cfg.get("tx_attenuation_db", 0))

    # Select Access Code
    ac_int = _ACCESS_CODE_INFO if args.mode == "info" else _ACCESS_CODE_JAMMER

    print("=" * 58)
    print("  2-GFSK Air-Link Test Transmitter")
    print("=" * 58)
    print(f"  Key       : {args.key}")
    print(f"  AC mode   : {args.mode}  (0x{ac_int:016X})")
    print(f"  Frequency : {tx_freq_hz / 1e6:.3f} MHz")
    print(f"  SR        : {_SAMPLE_RATE / 1e6:.2f} MSPS  SPS={_SPS}  BT={_BT}")
    print(f"  Deviation : {_DEVIATION_HZ / 1e3:.0f} kHz")
    print(f"  Pluto URI : {tx_uri}")
    print(f"  TX Atten  : {tx_atten} dB")
    print(f"  Repeats   : {args.repeats}×")
    print("=" * 58)

    # ── Build air-packet IQ ─────────────────────────────────────────────────
    frame = _build_rm_frame(args.key)
    print(f"\n[BUILD] RM Frame ({len(frame)} B): {frame.hex(' ').upper()}")

    # Slice into 15-byte payload chunks (pad last if needed)
    all_bits: list[int] = []
    for offset in range(0, len(frame), 15):
        chunk = frame[offset:offset + 15]
        if len(chunk) < 15:
            chunk = chunk + b"\x00" * (15 - len(chunk))
        pkt_bits = _build_air_packet_bits(chunk, ac_int)
        all_bits.extend(pkt_bits)

    print(f"[BUILD] Air-packet bits: {len(all_bits)}  "
          f"({len(all_bits) // 216} packet(s) × 216 bits)")

    # Modulate
    iq_frame = gfsk2_modulate_bits(
        all_bits,
        sps=_SPS,
        bt=_BT,
        span=_SPAN,
        deviation_hz=_DEVIATION_HZ,
        sample_rate=_SAMPLE_RATE,
    )
    print(f"[MOD]  IQ samples: {len(iq_frame)}  "
          f"({len(iq_frame) / _SAMPLE_RATE * 1000:.1f} ms)")

    # Repeat for glitch-free cyclic DMA
    iq_tx = np.tile(iq_frame, args.repeats).astype(np.complex64)
    # Scale to Pluto DAC: AD9363 12-bit, DMA 16-bit, −6 dBFS headroom
    dac_scale = float(2**14 * 0.5)
    iq_tx = (iq_tx * dac_scale).astype(np.complex64)
    print(f"[BUF]  Cyclic buffer: {len(iq_tx)} samples  "
          f"({len(iq_tx) / _SAMPLE_RATE * 1000:.1f} ms loop)")

    # ── Open Pluto and transmit ──────────────────────────────────────────────
    try:
        import adi
    except ImportError:
        print("\n[ERROR] pyadi-iio not installed.  Run: pip install pyadi-iio")
        sys.exit(1)

    print(f"\n[PLUTO] Connecting to {tx_uri} …")
    sdr = adi.Pluto(uri=tx_uri)

    # ── Silence TX before reconfiguration ───────────────────────────────────
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    sdr.tx_lo                  = tx_freq_hz
    sdr.sample_rate            = int(_SAMPLE_RATE)
    sdr.tx_rf_bandwidth        = int(_SAMPLE_RATE)
    sdr.tx_hardwaregain_chan0  = -abs(tx_atten)
    sdr.tx_cyclic_buffer       = True
    sdr.tx(iq_tx)

    print(f"[PLUTO] TX running — {tx_freq_hz / 1e6:.3f} MHz, "
          f"cyclic ({len(iq_tx)} samples)")
    print("[INFO]  Press Ctrl-C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        print("\n[PLUTO] TX stopped.")


if __name__ == "__main__":
    main()
