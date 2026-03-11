"""
main.py
=======
Entry Point — RCS-Radar-SDR  RM2026 Arena System

Wires all five modules together and launches the assessment dashboard.

Usage
─────
  python main.py [--config path/to/config.json] [--tx-only] [--rx-only]
                 [--key MYKEY] [--no-gui] [--demo]

Flags
─────
  --config  PATH   path to config.json  (default: ./config.json)
  --tx-only        only run Pluto TX; no RX / DSP
  --rx-only        only run RX + DSP; skip TX  (external transmitter assumed)
  --key     STR    override jammer key string  (default: from config or "RM2026")
  --no-gui         headless mode; decoded frames printed to stdout only
  --demo           use synthetic data (no hardware); GUI smoke-test
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCS-Radar-SDR RM2026 Arena System")
    p.add_argument("--config",   default=None,     metavar="PATH")
    p.add_argument("--tx-only",  action="store_true")
    p.add_argument("--rx-only",  action="store_true")
    p.add_argument("--key",      default="RM2026",  metavar="KEY")
    p.add_argument("--no-gui",   action="store_true")
    p.add_argument("--demo",     action="store_true")
    return p.parse_args()


def _on_frame(frame: dict, dashboard=None) -> None:
    """Common frame callback: print to stdout + forward to GUI if available."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {frame}")
    if dashboard is not None:
        dashboard.push_decoded(frame)


def main() -> None:
    args = _parse_args()

    # ── Demo mode (no hardware) ────────────────────────────────────────────
    if args.demo:
        from visual_terminal import Dashboard
        dash = Dashboard(demo_mode=True)
        dash.start()    # blocks
        return

    # ── Auto-detect PlutoSDR RX ────────────────────────────────────────────
    if not args.tx_only:
        from pathlib import Path
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
        from tx_signal_produce import PlutoTxProducer
        tx = PlutoTxProducer(mgr, key=args.key, simulate_arena=False)
        tx.start()
        print(f"[TX] Broadcasting key='{args.key}' on {plan.center_freq_hz / 1e6:.3f} MHz")

    # ── RX + DSP ───────────────────────────────────────────────────────────
    rx  = None
    dsp = None
    if not args.tx_only:
        from rx_sdr_driver  import PlutoRxDriver
        from dsp_processor  import DSPProcessor

        def on_frame(frame: dict) -> None:
            _on_frame(frame, dashboard)

        rx  = PlutoRxDriver(mgr, iq_queue)
        dsp = DSPProcessor(
            config    = mgr,
            in_queue  = iq_queue,
            on_frame  = on_frame,
            out_iq_q  = (dashboard._q_iq  if dashboard else None),
            out_sym_q = (dashboard._q_sym if dashboard else None),
        )
        t_dsp = threading.Thread(target=dsp.run_forever, daemon=True, name="DSP")
        t_dsp.start()
        rx.start()
        print(f"[RX] Listening on {plan.center_freq_hz / 1e6:.3f} MHz  "
              f"(gain={mgr.rx_gain_db} dB, channelise={plan.channelize})")

    # ── GUI event loop (blocks until window is closed) ─────────────────────
    if dashboard is not None:
        try:
            dashboard.start()       # returns only when user closes the window
        finally:
            _shutdown(rx, dsp, tx)
    else:
        # Headless: run until Ctrl-C
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
