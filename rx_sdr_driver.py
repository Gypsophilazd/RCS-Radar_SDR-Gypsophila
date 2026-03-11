"""
rx_sdr_driver.py
================
Module 3 — RX SDR Driver  (PlutoSDR / pyadi-iio)
RCS-Radar-SDR  RM2026 Arena System

Replaces the legacy pyrtlsdr backend.  Captures IQ from ADALM-PLUTO,
configures the AD9363 RX front-end, and pushes raw complex64 chunks
into a thread-safe Queue consumed by dsp_processor.py.

Why Manual Gain Control (MGC)?
──────────────────────────────
The arena contains a +10 dBm jammer that is ~70 dB above the −60 dBm
broadcast.  The AD9363 AGC tracks the *loudest* signal; if allowed to
run, it will set gain for the jammer and clip the broadcast completely.

MGC fixes the gain so both signals sit within the 12-bit ADC dynamic
range simultaneously.  Starting point: 20–30 dB.  Verify with
--diagnose (target IQ RMS 0.05–0.30 after the digital jammer filter).

Usage::

    from config_manager import load_config
    from rx_sdr_driver  import PlutoRxDriver
    import queue

    q   = queue.Queue(maxsize=8)
    mgr = load_config()
    rx  = PlutoRxDriver(mgr, q)
    rx.start()          # begins background thread
    ...
    rx.stop()
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config_manager import ConfigManager

# ── optional import guard ───────────────────────────────────────────────────
try:
    import adi  # pyadi-iio
    _PLUTO_AVAILABLE = True
except ImportError:
    _PLUTO_AVAILABLE = False


class PlutoRxDriver:
    """
    Wraps a PlutoSDR in RX-only mode and streams IQ into *out_queue*.

    Parameters
    ----------
    config  : loaded ConfigManager
    out_queue : queue.Queue[np.ndarray]  — receives complex64 arrays of
                length config.rx_buf_size
    """

    def __init__(self, config: "ConfigManager", out_queue: "queue.Queue[np.ndarray]"):
        self._cfg   = config
        self._queue = out_queue
        self._sdr   = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── public ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open PlutoSDR and start the background capture thread."""
        if not _PLUTO_AVAILABLE:
            raise RuntimeError("pyadi-iio not installed.  Run: pip install pyadi-iio")
        try:
            self._sdr = self._open_pluto()
        except Exception as exc:
            try:
                from config_manager import _pluto_diagnose_hint
                print(_pluto_diagnose_hint())
            except ImportError:
                pass
            raise RuntimeError(
                f"Cannot open PlutoSDR RX at URI '{self._cfg.pluto_rx_uri}'.\n"
                f"  Original error: {exc}\n"
                "  Run 'python config_manager.py' to scan for devices, or\n"
                "  manually set 'pluto_rx_uri' in config.json (e.g. \"usb:\")"
            ) from exc
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to terminate and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._sdr is not None:
            try:
                self._sdr.rx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None

    # ── private ──────────────────────────────────────────────────────────────

    def _open_pluto(self):
        """
        Configure and return an adi.Pluto instance.

        AD9363 RX gain control modes
        ─────────────────────────────
        "manual"       → rx_hardwaregain_chan0 sets gain directly  ← RECOMMENDED
        "slow_attack"  → AGC optimised for slowly varying signals
        "fast_attack"  → AGC optimised for bursty signals

        We force "manual" so that a +10 dBm nearby jammer cannot drive the
        AGC into a state that clips the −60 dBm broadcast.
        """
        plan = self._cfg.plan
        sdr  = adi.Pluto(uri=self._cfg.pluto_rx_uri)

        # ── RX configuration ──────────────────────────────────────────────
        sdr.rx_lo                     = int(plan.center_freq_hz)
        sdr.sample_rate               = int(plan.sample_rate_hz)
        sdr.rx_rf_bandwidth           = int(plan.sample_rate_hz)   # match SR
        sdr.rx_buffer_size            = self._cfg.rx_buf_size
        sdr.gain_control_mode_chan0   = "manual"
        sdr.rx_hardwaregain_chan0     = self._cfg.rx_gain_db

        # Disable TX so no accidental emission from the RX board
        sdr.tx_hardwaregain_chan0     = -89   # minimum, ~= muted

        return sdr

    def _capture_loop(self) -> None:
        """Background thread: pull IQ buffers and feed the queue."""
        while not self._stop_event.is_set():
            try:
                raw = self._sdr.rx()          # returns complex64 numpy array
                iq  = np.asarray(raw, dtype=np.complex64)
                # Normalize to float range approx ±1 (AD9363 outputs ±2^11)
                iq  = iq / (2.0 ** 11)
                try:
                    self._queue.put_nowait(iq)
                except queue.Full:
                    pass    # drop oldest block; DSP processor is behind
            except Exception as exc:
                print(f"[PlutoRxDriver] capture error: {exc}")
                time.sleep(0.1)


# ─── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from config_manager import load_config

    q   = queue.Queue(maxsize=4)
    mgr = load_config()
    print(mgr.plan.summary())
    print("Starting PlutoRxDriver for 3 seconds (press Ctrl-C to abort)…")
    rx = PlutoRxDriver(mgr, q)
    rx.start()
    time.sleep(3.0)
    rx.stop()
    print(f"Captured {q.qsize()} blocks in queue.")
    if not q.empty():
        chunk = q.get()
        rms = float(np.sqrt(np.mean(np.abs(chunk) ** 2)))
        print(f"First block: {len(chunk)} samples, IQ RMS = {rms:.4f}")
