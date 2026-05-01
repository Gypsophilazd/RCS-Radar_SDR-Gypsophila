#!/usr/bin/env python3
"""
tx_gfsk2_test.py
================
Standalone 2-GFSK Transmitter — RM2026 Air-Link Test Signal Generator.

Builds RM protocol frames, modulates via the shared phy/gfsk2_tx_builder,
and transmits continuously via ADALM-PLUTO in cyclic-buffer mode.

Usage
─────
  python3 tx_gfsk2_test.py --test-tx-enable --key RM2026
  python3 tx_gfsk2_test.py --test-tx-enable --key RM2026 --mode jammer
  python3 tx_gfsk2_test.py --test-tx-enable --freq 433.92 --attenuation 30

Cross-platform
──────────────
  Copy phy/ + packet_decoder.py + tx_gfsk2_test.py to the TX machine.
  RF signals are OS-independent — same PHY parameters on both ends.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from packet_decoder import crc8_rm, crc16_rm, SOF
from phy.gfsk2_tx_builder import (
    build_gfsk2_tx_iq,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPS,
    DEFAULT_BT,
    DEFAULT_SPAN,
    AC_INFO,
    AC_JAMMER,
)

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
                   help="TX frequency in MHz (default: from config team_color)")
    p.add_argument("--mode", default="info", choices=["info", "jammer"],
                   help="Access Code type: info or jammer (default: info)")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="Path to config.json")
    p.add_argument("--uri", default=None, metavar="URI",
                   help="Pluto URI (default: from config.json pluto_uri)")
    p.add_argument("--attenuation", type=float, default=None, metavar="dB",
                   help="TX attenuation 0..89 dB (default: from config.json)")
    p.add_argument("--repeats", type=int, default=4, metavar="N",
                   help="Frame repetitions in cyclic buffer (default: 4)")
    return p.parse_args()


def _load_config(config_path: str | None) -> dict:
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
    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, 1])
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + payload
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


def main() -> None:
    args = _parse_args()

    # ── Safety interlock ─────────────────────────────────────────────────────
    if not args.test_tx_enable:
        raise RuntimeError(
            "TX is disabled by default. "
            "Use --test-tx-enable only in legal test conditions."
        )

    # ── Load config for PHY parameters ────────────────────────────────────────
    cfg = _load_config(args.config)
    our_color = str(cfg.get("team_color", "red")).lower().strip()

    # Load deviation from PhyConfig if available, else default
    try:
        from config_manager import load_config as load_mgr
        mgr = load_mgr(args.config)
        deviation_hz = mgr.phy_config.deviation_hz
    except Exception:
        deviation_hz = 250_000.0

    # Determine frequency
    if args.freq is not None:
        tx_freq_hz = int(args.freq * 1e6)
    else:
        tx_freq_hz = int(_BROADCAST_FREQ.get(our_color, 433_200_000))

    # Determine URI
    tx_uri = args.uri or str(cfg.get("pluto_uri", "ip:192.168.2.1"))

    # Determine attenuation
    tx_atten = (args.attenuation if args.attenuation is not None
                else int(cfg.get("tx_attenuation_db", 0)))

    print("=" * 58)
    print("  2-GFSK Air-Link Test Transmitter")
    print("=" * 58)
    print(f"  Key        : {args.key}")
    print(f"  AC mode    : {args.mode}")
    print(f"  Frequency  : {tx_freq_hz / 1e6:.3f} MHz")
    print(f"  SR         : {DEFAULT_SAMPLE_RATE / 1e6:.2f} MSPS"
          f"  SPS={DEFAULT_SPS}  BT={DEFAULT_BT}")
    print(f"  Deviation  : {deviation_hz / 1e3:.1f} kHz")
    print(f"  Pluto URI  : {tx_uri}")
    print(f"  TX Atten   : {tx_atten} dB")
    print(f"  Repeats    : {args.repeats}×")
    print("=" * 58)

    # ── Build frame and modulate ────────────────────────────────────────────
    frame = _build_rm_frame(args.key)
    print(f"\n[BUILD] RM Frame ({len(frame)} B): {frame.hex(' ').upper()}")

    iq_tx = build_gfsk2_tx_iq(
        frame,
        mode=args.mode,
        deviation_hz=deviation_hz,
        sample_rate=DEFAULT_SAMPLE_RATE,
        sps=DEFAULT_SPS,
        bt=DEFAULT_BT,
        span=DEFAULT_SPAN,
        repeats=args.repeats,
    )
    n_packets = (len(frame) + 14) // 15
    print(f"[BUILD] {n_packets} air packet(s), "
          f"total bits: {n_packets * 216}")
    print(f"[MOD]   IQ samples: {len(iq_tx)}  "
          f"({len(iq_tx) / DEFAULT_SAMPLE_RATE * 1000:.1f} ms loop)")

    # ── Open Pluto and transmit ──────────────────────────────────────────────
    try:
        import adi
    except ImportError:
        print("\n[ERROR] pyadi-iio not installed.  Run: pip install pyadi-iio")
        sys.exit(1)

    print(f"\n[PLUTO] Connecting to {tx_uri} …")
    sdr = adi.Pluto(uri=tx_uri)

    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    sdr.tx_lo                 = tx_freq_hz
    sdr.sample_rate           = int(DEFAULT_SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(DEFAULT_SAMPLE_RATE)
    sdr.tx_hardwaregain_chan0 = -abs(tx_atten)
    sdr.tx_cyclic_buffer      = True
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
