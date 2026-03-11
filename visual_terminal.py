"""
visual_terminal.py
==================
Module 5 — Assessment Dashboard GUI
RCS-Radar-SDR  RM2026 Arena System

3-Panel real-time dashboard powered by pyqtgraph.

Panel 1 — Spectrum Analyser (Raw RF)
    FFT of incoming IQ, averaged over N frames.
    Vertical markers at broadcast and jammer frequencies.

Panel 2 — DSP Pipeline Output
    Left  : Symbol scatter  —  symbol index vs recovered level {-3,-1,+1,+3}.
    Right : Symbol histogram  —  distribution across the four levels.
    Slicer threshold lines at -2, 0, +2.

Panel 3 — Decoded Information Table
    Real-time table: newest row on top.
    Columns: Timestamp | CmdID | Name | Payload
    Gold highlight for 0x0A06 (jammer key), green for 0x0A01 (positions).

Thread-safe injection API (call from ANY thread):
    dashboard.push_iq(iq, centre_hz, sr_hz)
    dashboard.push_symbols(symbols)
    dashboard.push_decoded(frame_dict)

Standalone smoke-test (synthetic data, no hardware):
    python visual_terminal.py

Integrated usage:
    from config_manager  import load_config
    from visual_terminal import Dashboard

    mgr  = load_config()
    dash = Dashboard(freq_plan=mgr.plan)
    dash.start()    # blocks until window is closed
"""

from __future__ import annotations

import queue
import sys
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    raise SystemExit(
        "pyqtgraph is required.  Run:  pip install pyqtgraph PyQt5"
    )

try:
    from config_manager import FreqPlan
except ImportError:
    FreqPlan = None  # type: ignore

# ─── Tunable constants ─────────────────────────────────────────────────────────

FFT_SIZE          = 2048
FFT_AVG_N         = 6           # exponential average over N frames (higher = smoother)
WAVEFORM_SAMPLES  = 2048        # IQ time-domain window: samples to display
SYMBOL_HISTORY    = 512         # scatter: keep last N symbols
TABLE_MAX_ROWS    = 300
UPDATE_MS         = 50          # QTimer interval → 20 Hz GUI refresh

# Colour palette
_C_SPECTRUM  = (64,  192, 255)
_C_BROADCAST = (80,  255, 100)
_C_JAMMER    = (255,  80,  80)
_C_GRID      = (55,   55,  70)
_C_I_WAVE    = (100, 180, 255)   # I trace: light blue
_C_Q_WAVE    = (255, 150,  50)   # Q trace: amber
_C_SYM = {
    -3: (255,  80,  80, 200),
    -1: (255, 200,  30, 200),
    +1: ( 80, 200, 255, 200),
    +3: (100, 255, 100, 200),
}

CMD_NAMES = {
    "0x0A01": "Enemy Positions",
    "0x0A02": "Enemy HP",
    "0x0A03": "Enemy Ammo",
    "0x0A04": "Macro Status",
    "0x0A05": "Enemy Buffs",
    "0x0A06": "Jammer Key ★",
}
CMD_COLOURS = {
    "0x0A06": "#FFD700",   # gold — most important
    "0x0A01": "#90EE90",   # green
    "0x0A02": "#FF9999",   # pink/red
    "0x0A04": "#87CEEB",   # sky blue
}
_DEFAULT_COLOUR = "#D0D0D0"

pg.setConfigOptions(antialias=True, background="#12122a", foreground="#e0e0e0")


# ─── Panel 1: Spectrum Analyser ────────────────────────────────────────────────

class SpectrumWidget(pg.PlotWidget):
    """
    Real-time power spectral density display.

    Features
    ─────────
    • Blackman-windowed FFT, exponentially averaged.
    • Absolute frequency axis (Hz) computed from centre_hz + SR.
    • InfiniteLine markers at broadcast and jammer frequencies with labels.
    • Y-axis in dBFS (0 dBFS = ADC full-scale equivalent).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._centre_hz = 433_920_000.0
        self._sr_hz     = 2_500_000
        self._avg: Optional[np.ndarray] = None

        self.setTitle("Panel 1 — Spectrum Analyser (Raw RF)", color="#64C0FF", size="10pt")
        self.setLabel("left",   "Power (dBFS)")
        self.setLabel("bottom", "Frequency (MHz)")
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setYRange(-90, 5)

        self._curve = self.plot(pen=pg.mkPen(_C_SPECTRUM, width=1))

        _dash = QtCore.Qt.PenStyle.DashLine
        self._mark_bc = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(_C_BROADCAST, width=1.5, style=_dash),
            label="Broadcast",
            labelOpts={"color": _C_BROADCAST, "position": 0.90, "anchors": [(0.5, 0), (0.5, 1)]},
        )
        self._mark_jam = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(_C_JAMMER, width=1.5, style=_dash),
            label="Jammer",
            labelOpts={"color": _C_JAMMER, "position": 0.75, "anchors": [(0.5, 0), (0.5, 1)]},
        )
        self.addItem(self._mark_bc)
        self.addItem(self._mark_jam)
        self._mark_jam.setVisible(False)

    def set_freq_plan(self, plan) -> None:
        self._mark_bc.setValue(plan.broadcast_freq_hz / 1e6)
        if plan.jammer_freq_hz is not None:
            self._mark_jam.setValue(plan.jammer_freq_hz / 1e6)
            self._mark_jam.setVisible(True)
        else:
            self._mark_jam.setVisible(False)

    def update_iq(self, iq: np.ndarray, centre_hz: float, sr_hz: int) -> None:
        self._centre_hz = centre_hz
        self._sr_hz     = sr_hz
        n   = min(len(iq), FFT_SIZE)
        win = np.blackman(n).astype(np.float32)
        seg = iq[:n] * win
        raw = np.abs(np.fft.fftshift(np.fft.fft(seg, n=FFT_SIZE))) ** 2
        raw = np.maximum(raw, 1e-15)
        psd = 10 * np.log10(raw / (np.sum(win ** 2) * raw.max()))

        alpha = 2.0 / (FFT_AVG_N + 1)
        if self._avg is None or self._avg.shape != psd.shape:
            self._avg = psd.copy()
        else:
            self._avg = alpha * psd + (1 - alpha) * self._avg

        freqs = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1.0 / sr_hz)) + centre_hz) / 1e6
        self._curve.setData(freqs, self._avg)


# ─── Panel 0: IQ Time-Domain Waveform ────────────────────────────────────────

class IQWaveformWidget(pg.PlotWidget):
    """
    Panel 0 — Raw IQ waveform (time domain).

    Displays the last ``WAVEFORM_SAMPLES`` samples split into I (real) and
    Q (imaginary) traces on a time axis in microseconds.  Amplitude is
    normalised to ±1 so the display is scale-invariant across gain settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Panel 0 — Raw IQ Waveform (Time Domain)", color="#64C0FF", size="10pt")
        self.setLabel("left",   "Amplitude (norm.)")
        self.setLabel("bottom", "Time (µs)")
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setYRange(-1.1, 1.1)
        self.addLegend(offset=(-10, 10))
        self._curve_i = self.plot(pen=pg.mkPen(_C_I_WAVE, width=1), name="I")
        self._curve_q = self.plot(pen=pg.mkPen(_C_Q_WAVE, width=1), name="Q")

    def update_iq(self, iq: np.ndarray, centre_hz: float, sr_hz: int) -> None:
        n     = min(len(iq), WAVEFORM_SAMPLES)
        seg   = iq[:n]
        t_us  = np.arange(n, dtype=np.float32) * 1e6 / sr_hz
        peak  = float(np.max(np.abs(seg))) or 1.0
        self._curve_i.setData(t_us, (seg.real / peak).astype(np.float32))
        self._curve_q.setData(t_us, (seg.imag / peak).astype(np.float32))


# ─── Panel 2: Symbol Scatter + Histogram ──────────────────────────────────────

class SymbolScatterWidget(pg.GraphicsLayoutWidget):
    """
    Left  : time-indexed symbol scatter with threshold lines.
    Right : symbol distribution bar chart.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#12122a")
        self._history: deque[float] = deque(maxlen=SYMBOL_HISTORY)

        # ── Left: scatter ──────────────────────────────────────────────────
        self._ps = self.addPlot(title="Panel 2 — Recovered Symbols (MF Output)")
        self._ps.setLabel("left",   "Symbol Level")
        self._ps.setLabel("bottom", "Symbol Index")
        self._ps.showGrid(x=True, y=True, alpha=0.18)
        self._ps.setYRange(-4.5, 4.5)

        self._series: dict[int, pg.ScatterPlotItem] = {}
        for sym, col in _C_SYM.items():
            sp = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(*col))
            self._ps.addItem(sp)
            self._series[sym] = sp

        _dot = QtCore.Qt.PenStyle.DotLine
        for thr in (-2.0, 0.0, 2.0):
            self._ps.addItem(pg.InfiniteLine(
                pos=thr, angle=0, movable=False,
                pen=pg.mkPen(_C_GRID, width=1, style=_dot),
            ))

        # ── Right: histogram ───────────────────────────────────────────────
        self._ph = self.addPlot(title="Symbol Distribution")
        self._ph.setLabel("left",   "Count")
        self._ph.setLabel("bottom", "Level")
        self._ph.setXRange(-4.2, 4.2)

        _bar_colours = [pg.mkBrush(*_C_SYM[s]) for s in (-3, -1, 1, 3)]
        self._bars = pg.BarGraphItem(
            x=[-3, -1, 1, 3], height=[0, 0, 0, 0], width=0.65,
            brushes=_bar_colours,
        )
        self._ph.addItem(self._bars)

        # percentage labels
        self._pct_labels: list[pg.TextItem] = []
        for xpos in (-3, -1, 1, 3):
            lbl = pg.TextItem("", color="#e0e0e0", anchor=(0.5, 1.0))
            lbl.setPos(xpos, 0)
            self._ph.addItem(lbl)
            self._pct_labels.append(lbl)

    def update_symbols(self, symbols: np.ndarray) -> None:
        self._history.extend(symbols.tolist())
        arr  = np.array(self._history, dtype=np.float32)
        idx  = np.arange(len(arr))
        sliced = np.where(arr < -2, -3, np.where(arr < 0, -1, np.where(arr < 2, 1, 3)))

        for sym, sp in self._series.items():
            mask = sliced == sym
            sp.setData(x=idx[mask], y=arr[mask]) if mask.any() else sp.setData([], [])

        total = max(len(arr), 1)
        counts = [int((sliced == s).sum()) for s in (-3, -1, 1, 3)]
        self._bars.setOpts(height=counts)
        for i, (cnt, lbl) in enumerate(zip(counts, self._pct_labels)):
            pct = 100.0 * cnt / total
            lbl.setText(f"{pct:.1f}%")
            lbl.setPos([-3, -1, 1, 3][i], cnt)


# ─── Panel 3: Decoded Information Table ───────────────────────────────────────

class DecodedTableWidget(QtWidgets.QWidget):
    """Newest on top, colour-coded by CmdID."""

    _HEADERS = ["Time", "CmdID", "Name", "Payload"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background-color:#12122a; color:#d0d0d0;"
            "font-family:'Consolas','Courier New',monospace; font-size:12px;"
        )
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 2)

        title = QtWidgets.QLabel("Panel 3 — Decoded Information")
        title.setStyleSheet("color:#64C0FF; font-size:13px; font-weight:bold; padding:2px;")
        vbox.addWidget(title)

        self._tbl = QtWidgets.QTableWidget(0, len(self._HEADERS))
        self._tbl.setHorizontalHeaderLabels(self._HEADERS)
        self._tbl.horizontalHeader().setStretchLastSection(True)
        self._tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setColumnWidth(0, 95)
        self._tbl.setColumnWidth(1, 70)
        self._tbl.setColumnWidth(2, 155)
        self._tbl.setStyleSheet(
            "QTableWidget{background:#0e0e24;alternate-background-color:#171730;"
            "  color:#d0d0d0;gridline-color:#2a2a44;}"
            "QHeaderView::section{background:#0a0a1e;color:#64C0FF;"
            "  border:1px solid #2a2a44;padding:4px;}"
            "QTableWidget::item:selected{background:#2a2a5a;}"
        )
        vbox.addWidget(self._tbl)

        self._status = QtWidgets.QLabel("Frames: 0  |  Last: —")
        self._status.setStyleSheet("color:#666;font-size:11px;padding:2px;")
        vbox.addWidget(self._status)

        self._count = 0

    def push_frame(self, frame: dict) -> None:
        cmd     = frame.get("cmd", "0x????")
        name    = CMD_NAMES.get(cmd, "Unknown")
        ts      = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        colour  = CMD_COLOURS.get(cmd, _DEFAULT_COLOUR)

        # Build compact payload string (skip meta fields)
        payload = "  ".join(
            f"{k}={v}" for k, v in frame.items() if k not in ("cmd", "seq")
        )

        self._tbl.insertRow(0)
        for col, text in enumerate([ts, cmd, name, payload]):
            item = QtWidgets.QTableWidgetItem(text)
            item.setForeground(pg.mkColor(colour))
            self._tbl.setItem(0, col, item)

        while self._tbl.rowCount() > TABLE_MAX_ROWS:
            self._tbl.removeRow(self._tbl.rowCount() - 1)

        self._count += 1
        self._status.setText(f"Frames: {self._count}  |  Last: {cmd} — {name}")


# ─── Demo worker (standalone smoke-test only) ──────────────────────────────────

class _DemoWorker(QtCore.QThread):
    """
    Generates synthetic radio + symbol data for the standalone smoke-test.
    Replace with PlutoRxDriver + DSPProcessor when wiring real hardware.
    """
    sig_iq      = QtCore.Signal(np.ndarray, float, int)
    sig_symbols = QtCore.Signal(np.ndarray)
    sig_frame   = QtCore.Signal(dict)

    def run(self):
        sr     = 2_500_000
        centre = 433_920_000.0
        seq    = 0
        tick   = 0
        rng    = np.random.default_rng(42)
        while not self.isInterruptionRequested():
            n       = 2048
            syms    = rng.choice(np.array([-3, -1, 1, 3]), size=n // 8)
            upsampl = np.repeat(syms, 8).astype(np.float32)
            freq    = upsampl * (250_000.0 / 3.0)
            phase   = np.cumsum(freq) * 2 * np.pi / sr
            iq      = (0.25 * np.exp(1j * phase) + 0.02 * (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)
            )).astype(np.complex64)
            # Synthetic +10 dBm jammer at +1 MHz
            t_arr = np.arange(n, dtype=np.float32) / sr
            iq += (0.85 * np.exp(1j * 2 * np.pi * 1_000_000.0 * t_arr)).astype(np.complex64)

            self.sig_iq.emit(iq, centre, sr)
            self.sig_symbols.emit(syms.astype(np.float32))

            if tick % 18 == 0:
                self.sig_frame.emit({"cmd": "0x0A06", "key": "RM2026", "seq": seq})
            if tick % 31 == 0:
                self.sig_frame.emit({
                    "cmd": "0x0A01", "seq": seq,
                    "positions": {
                        "hero":        (round(float(rng.uniform(2, 26)), 2), round(float(rng.uniform(1, 14)), 2)),
                        "infantry_3":  (round(float(rng.uniform(2, 26)), 2), round(float(rng.uniform(1, 14)), 2)),
                    },
                    "valid_mask": "0x001F",
                })
            if tick % 53 == 0:
                self.sig_frame.emit({
                    "cmd": "0x0A02", "seq": seq,
                    "hp": {"hero": 600, "engineer": 400, "infantry_3": 150,
                           "infantry_4": 250, "infantry_5": 0, "sentry": 800},
                })

            tick += 1
            seq  += 1
            QtCore.QThread.msleep(50)


# ─── Dashboard (top-level controller) ─────────────────────────────────────────

class Dashboard:
    """
    Creates the Qt application, wires the three panels, and manages
    a QTimer-driven queue-drain loop.

    Parameters
    ----------
    freq_plan  : FreqPlan from config_manager (optional; enables frequency markers)
    demo_mode  : if True, starts _DemoWorker and runs with synthetic data
    """

    def __init__(self, freq_plan=None, demo_mode: bool = False):
        self._app  = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self._plan = freq_plan
        self._demo = demo_mode

        # Thread-safe queues — RX driver / DSP processor writes, GUI timer reads
        self._q_iq:  "queue.Queue[tuple]"  = queue.Queue(maxsize=6)
        self._q_sym: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        self._q_dec: "queue.Queue[dict]"       = queue.Queue(maxsize=128)

        self._build_window()

        self._timer = QtCore.QTimer()
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._poll)

        if self._plan is not None:
            self._spectrum.set_freq_plan(self._plan)

    # ── public injection API (thread-safe) ───────────────────────────────────

    def push_iq(self, iq: np.ndarray, centre_hz: float, sr_hz: int) -> None:
        """Call from any thread; GUI will pick it up at next tick."""
        try:
            self._q_iq.put_nowait((iq, centre_hz, sr_hz))
        except queue.Full:
            pass

    def push_symbols(self, symbols: np.ndarray) -> None:
        try:
            self._q_sym.put_nowait(symbols)
        except queue.Full:
            pass

    def push_decoded(self, frame: dict) -> None:
        try:
            self._q_dec.put_nowait(frame)
        except queue.Full:
            pass

    def start(self) -> None:
        """Show window, start demo worker if requested, then run Qt event loop."""
        self._win.show()
        self._timer.start()

        if self._demo:
            self._worker = _DemoWorker()
            self._worker.sig_iq.connect(self.push_iq)
            self._worker.sig_symbols.connect(self.push_symbols)
            self._worker.sig_frame.connect(self.push_decoded)
            self._worker.start()

        sys.exit(self._app.exec())

    # ── private ──────────────────────────────────────────────────────────────

    def _build_window(self) -> None:
        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(
            "RCS-Radar-SDR  │  RM2026 Arena Assessment Dashboard"
        )
        self._win.resize(1400, 1100)
        self._win.setStyleSheet("background:#12122a;")

        central = QtWidgets.QWidget()
        self._win.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setSpacing(3)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Title bar ─────────────────────────────────────────────────────
        hdr = QtWidgets.QLabel(
            "🛰  RCS-Radar-SDR  │  RM2026 Electronic Warfare — Assessment Dashboard"
        )
        hdr.setStyleSheet(
            "font-size:14px;font-weight:bold;color:#64C0FF;"
            "padding:5px 8px;background:#0a0a1e;border-radius:4px;"
        )
        root.addWidget(hdr)

        # ── Splitter: panel 1 + 2 side-by-side (top), panel 3 (bottom) ───
        v_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top row: spectrum (left) + symbol scatter (right)
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_widget)
        top_layout.setSpacing(4)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Left column: spectrum (top) + IQ waveform (bottom)
        left_col  = QtWidgets.QWidget()
        left_vbox = QtWidgets.QVBoxLayout(left_col)
        left_vbox.setSpacing(3)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        self._spectrum = SpectrumWidget()
        self._waveform = IQWaveformWidget()
        left_vbox.addWidget(self._spectrum, stretch=1)
        left_vbox.addWidget(self._waveform, stretch=1)

        self._scatter  = SymbolScatterWidget()
        top_layout.addWidget(left_col,       stretch=5)
        top_layout.addWidget(self._scatter,  stretch=4)
        top_widget.setMinimumHeight(420)
        v_split.addWidget(top_widget)

        # Bottom: decoded table
        self._table = DecodedTableWidget()
        self._table.setMinimumHeight(220)
        v_split.addWidget(self._table)

        v_split.setSizes([440, 260])
        root.addWidget(v_split)

        # ── Status bar ────────────────────────────────────────────────────
        freq_txt = ""
        if self._plan is not None:
            freq_txt = (
                f"  Centre: {self._plan.center_freq_hz / 1e6:.3f} MHz  │"
                f"  SR: {self._plan.sample_rate_hz / 1e6:.2f} MSPS  │"
                f"  Channelise: {self._plan.channelize}"
            )
        sb = self._win.statusBar()
        sb.setStyleSheet("background:#0a0a1e;color:#666;font-size:11px;")
        sb.showMessage(f"RM2026 Arena Mode{freq_txt}  │  Refresh: {UPDATE_MS} ms")

    def _poll(self) -> None:
        """Drain all three queues and refresh panels (called 20×/s in GUI thread)."""
        # Spectrum: keep only the latest IQ block
        iq_item = None
        while not self._q_iq.empty():
            iq_item = self._q_iq.get_nowait()
        if iq_item is not None:
            if isinstance(iq_item, tuple) and len(iq_item) == 3:
                iq_args = iq_item
            else:
                centre_hz = self._plan.center_freq_hz if self._plan is not None else 0.0
                sr_hz = self._plan.sample_rate_hz if self._plan is not None else 2_500_000
                iq_args = (iq_item, centre_hz, sr_hz)
            self._spectrum.update_iq(*iq_args)
            self._waveform.update_iq(*iq_args)

        # Symbols: batch-accumulate all pending vectors
        batches: list[np.ndarray] = []
        while not self._q_sym.empty():
            batches.append(self._q_sym.get_nowait())
        if batches:
            self._scatter.update_symbols(np.concatenate(batches))

        # Decoded frames: insert all pending rows
        while not self._q_dec.empty():
            self._table.push_frame(self._q_dec.get_nowait())


# ─── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    _dash = Dashboard(demo_mode=True)
    _dash.start()
