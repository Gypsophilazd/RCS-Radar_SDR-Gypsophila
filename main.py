"""
main.py
=======
Entry Point — RCS-Radar-SDR  RM2026 Arena System

Wires all modules together and launches the assessment dashboard.

Usage
─────
  python main.py [--config path/to/config.json] [--tx-only] [--rx-only]
                 [--key MYKEY] [--air-mode info|jammer] [--freq MHz]
                 [--no-gui] [--demo] [--test-tx-enable]

Flags
─────
  --config    PATH   path to config.json  (default: ./config.json)
  --tx-only          only run Pluto TX; no RX / DSP
  --rx-only          only run RX + DSP; skip TX
  --key       STR    override jammer key string  (default: "RM2026")
  --air-mode  STR    Air Packet Access Code: "info" or "jammer" (default: "info")
  --freq      MHz    TX frequency override in MHz (default: auto from config)
  --no-gui           headless mode; decoded frames printed to stdout only
  --demo             use synthetic data (no hardware); GUI smoke-test
  --test-tx-enable   REQUIRED to enable TX (safety interlock)
"""

from __future__ import annotations

import argparse
import json
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from packet_decoder import crc8_rm, crc16_rm, SOF


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCS-Radar-SDR RM2026 Arena System")
    p.add_argument("--config",   default=None,     metavar="PATH")
    p.add_argument("--tx-only",  action="store_true")
    p.add_argument("--rx-only",  action="store_true")
    p.add_argument("--key",      default="RM2026",  metavar="KEY")
    p.add_argument("--air-mode", default="info",   choices=["info", "jammer"],
                   help="Air Packet Access Code type (default: info)")
    p.add_argument("--jammer-level", type=int, default=0, choices=[0, 1, 2, 3],
                   help="Jammer level 1-3 (TX uses jammer freq/deviation)")
    p.add_argument("--freq",     type=float, default=None, metavar="MHz",
                   help="TX frequency override in MHz (deviation follows --jammer-level)")
    p.add_argument("--rx-source", default=None, choices=["broadcast", "jammer"],
                   help="RX listen target (default: from config.json rx_source)")
    p.add_argument("--rx-freq", type=float, default=None, metavar="MHz",
                   help="RX direct-tune frequency override at 1 MSPS (bypasses channelizer)")
    p.add_argument("--no-gui",   action="store_true")
    p.add_argument("--demo",     action="store_true")
    p.add_argument("--test-tx-enable", action="store_true",
                   help="Enable TX (disabled by default)")
    return p.parse_args()


def _on_frame(frame: dict, dashboard=None) -> None:
    """Common frame callback: print to stdout + forward to GUI if available."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {frame}")
    if dashboard is not None:
        dashboard.push_decoded(frame)


def _build_rm_frame(key: str) -> bytes:
    """Build a valid 0x0A06 RM frame with CRC8/CRC16."""
    cmd_id = 0x0A06
    key_bytes = key.encode("ascii")[:6].ljust(6, b"\x00")
    payload = struct.pack("<H", cmd_id) + key_bytes
    data_len = len(payload)
    header = bytes([SOF, data_len & 0xFF, (data_len >> 8) & 0xFF, 1])
    crc8_val = crc8_rm(header)
    crc16_input = header + bytes([crc8_val]) + payload
    crc16_val = crc16_rm(crc16_input)
    return crc16_input + struct.pack("<H", crc16_val)


class _Gfsk2Tx:
    """Thin wrapper around the 2-GFSK TX builder + Pluto SDR."""

    def __init__(self, uri: str, freq_hz: int, atten_db: float,
                 sr: float, deviation_hz: float, bt: float, sps: int):
        self._uri = uri
        self._freq_hz = freq_hz
        self._atten = atten_db
        self._sr = sr
        self._deviation = deviation_hz
        self._bt = bt
        self._sps = sps
        self._sdr = None

    def start(self, frame: bytes, air_mode: str) -> None:
        from phy.gfsk2_tx_builder import build_gfsk2_tx_iq
        import adi

        iq_tx = build_gfsk2_tx_iq(
            frame, mode=air_mode,
            deviation_hz=self._deviation,
            sample_rate=self._sr,
            sps=self._sps, bt=self._bt,
        )

        self._sdr = adi.Pluto(uri=self._uri)
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:
            pass
        self._sdr.tx_lo                 = self._freq_hz
        self._sdr.sample_rate           = int(self._sr)
        self._sdr.tx_rf_bandwidth       = int(self._sr)
        self._sdr.tx_hardwaregain_chan0 = -abs(self._atten)
        self._sdr.tx_cyclic_buffer      = True
        self._sdr.tx(iq_tx)

    def stop(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None


def main() -> None:
    args = _parse_args()

    # ── Demo mode (no hardware) ────────────────────────────────────────────
    if args.demo:
        from visual_terminal import Dashboard
        dash = Dashboard(demo_mode=True)
        dash.start()
        return

    # ── Auto-detect PlutoSDR RX ────────────────────────────────────────────
    if not args.tx_only:
        from config_manager import detect_pluto_rx_uri, save_rx_uri_to_config

        config_path = Path(args.config) if args.config else Path(__file__).parent / "config.json"
        explicit_rx_uri = None
        try:
            raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            explicit_rx_uri = str(raw_cfg.get("pluto_rx_uri", "")).strip()
        except Exception:
            explicit_rx_uri = None

        should_auto_detect = explicit_rx_uri in (None, "", "auto", "usb:")
        if should_auto_detect:
            print("[AUTO] Scanning for PlutoSDR RX … ", end="", flush=True)
            _uri = detect_pluto_rx_uri()
            if _uri:
                print(f"found {_uri}")
                save_rx_uri_to_config(args.config, _uri)
            else:
                from config_manager import _pluto_diagnose_hint
                print("not found — using value from config.json")
                print(_pluto_diagnose_hint())
        else:
            print(f"[AUTO] Using configured Pluto RX URI: {explicit_rx_uri}")

    # ── Load configuration ─────────────────────────────────────────────────
    from config_manager import load_config
    mgr  = load_config(args.config)
    plan = mgr.plan
    phy  = mgr.phy_config
    print(plan.summary())

    # ── Queues ─────────────────────────────────────────────────────────────
    iq_queue:  queue.Queue = queue.Queue(maxsize=8)
    sym_queue: queue.Queue = queue.Queue(maxsize=16)

    # ── Dashboard ──────────────────────────────────────────────────────────
    dashboard = None
    if not args.no_gui:
        from visual_terminal import Dashboard
        dashboard = Dashboard(freq_plan=plan, demo_mode=False)

    # ── TX ─────────────────────────────────────────────────────────────────
    tx = None
    if not args.rx_only:
        if not args.test_tx_enable:
            raise RuntimeError(
                "TX is disabled by default. "
                "Use --test-tx-enable only in legal test conditions."
            )

        # Dispatch by PHY mode
        if phy.mode == "4rrcfsk_legacy":
            from legacy_tx_signal_produce import PlutoTxProducer
            tx = PlutoTxProducer(mgr, key=args.key, simulate_arena=False,
                                 test_tx_enabled=True)
            tx.start()
            print(f"[TX] Legacy 4-RRC-FSK: key='{args.key}' on "
                  f"{plan.our_broadcast_freq_hz / 1e6:.3f} MHz")
        else:
            # Default: 2-GFSK air-packet TX
            jlevel = args.jammer_level

            # Determine TX source: broadcast or jammer
            is_jammer_tx = (args.air_mode == "jammer" and jlevel > 0)
            tx_source = "jammer" if is_jammer_tx else "broadcast"

            # Frequency: --freq overrides; otherwise use official freq
            from config_manager import get_broadcast_frequency, get_jammer_frequency
            if args.freq is not None:
                tx_freq_hz = int(args.freq * 1e6)
            elif is_jammer_tx:
                tx_freq_hz = int(get_jammer_frequency(plan.our_color, jlevel))
            else:
                tx_freq_hz = int(plan.our_broadcast_freq_hz)

            # Deviation: follows jammer level if jammer TX, else broadcast
            if is_jammer_tx:
                tx_dev = phy.jammer_deviation_hz
            else:
                tx_dev = phy.deviation_hz

            tx = _Gfsk2Tx(
                uri=mgr.pluto_uri,
                freq_hz=tx_freq_hz,
                atten_db=mgr.tx_attenuation_db,
                sr=phy.sample_rate,
                deviation_hz=tx_dev,
                bt=phy.bt,
                sps=phy.sps,
            )
            frame = _build_rm_frame(args.key)
            tx.start(frame, air_mode=args.air_mode)

            print("=" * 58)
            print("  2-GFSK Air-Link TX")
            print("=" * 58)
            print(f"  TX source  : {tx_source}")
            print(f"  Team color : {plan.our_color.upper()}")
            print(f"  Frequency  : {tx_freq_hz / 1e6:.3f} MHz")
            print(f"  Air mode   : {args.air_mode}")
            print(f"  Jammer lvl : {jlevel}")
            print(f"  Sample rate: {phy.sample_rate / 1e6:.2f} MSPS")
            print(f"  SPS        : {phy.sps}")
            print(f"  BT         : {phy.bt}")
            print(f"  Deviation  : {tx_dev / 1e3:.1f} kHz")
            print(f"  TX Atten   : {mgr.tx_attenuation_db} dB")
            print(f"  Pluto URI  : {mgr.pluto_uri}")
            print(f"  Key        : {args.key}")
            print(f"  Frame      : {frame.hex(' ').upper()}")
            print("=" * 58)

    # ── RX + DSP ───────────────────────────────────────────────────────────
    rx  = None
    dsp = None
    if not args.tx_only:
        from rx_sdr_driver  import PlutoRxDriver
        from dsp_processor  import DSPProcessor

        # Resolve rx_source: CLI overrides config.json
        rx_src = args.rx_source if args.rx_source is not None else mgr.rx_source
        if rx_src == "jammer" and plan.jammer_level == 0:
            raise RuntimeError(
                "rx_source='jammer' requires target_jammer_level > 0. "
                "Set target_jammer_level in config.json to 1, 2, or 3."
            )

        # --rx-freq: direct-tune at 1 MSPS, bypass channelizer
        rx_freq_override = None
        rx_sr_override = None
        if args.rx_freq is not None:
            rx_freq_override = int(args.rx_freq * 1e6)
            rx_sr_override = 1_000_000.0

        def on_frame(frame: dict) -> None:
            _on_frame(frame, dashboard)

        rx  = PlutoRxDriver(mgr, iq_queue,
                            freq_hz=rx_freq_override,
                            sample_rate_hz=rx_sr_override)
        dsp = DSPProcessor(
            config    = mgr,
            in_queue  = iq_queue,
            on_frame  = on_frame,
            out_iq_q  = (dashboard._q_iq  if dashboard else None),
            out_sym_q = (dashboard._q_sym if dashboard else None),
            rx_source = rx_src,
            direct_tune=(args.rx_freq is not None),
        )
        t_dsp = threading.Thread(target=dsp.run_forever, daemon=True, name="DSP")
        t_dsp.start()
        rx.start()
        rx_freq_mhz = (rx_freq_override / 1e6 if rx_freq_override
                       else plan.center_freq_hz / 1e6)
        print(f"[RX] Listening on {rx_freq_mhz:.3f} MHz  "
              f"(gain={mgr.rx_gain_db} dB, channelise={plan.channelize}, "
              f"source={rx_src})")

    # ── GUI event loop ─────────────────────────────────────────────────────
    if dashboard is not None:
        try:
            dashboard.start()
        finally:
            _shutdown(rx, dsp, tx)
    else:
        print("[INFO] Headless mode — press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(rx, dsp, tx)


def _shutdown(rx, dsp, tx) -> None:
    print("[INFO] Shutting down…")
    if rx  is not None: rx.stop()
    if dsp is not None: dsp.stop()
    if tx  is not None: tx.stop()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
