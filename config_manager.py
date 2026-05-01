"""
config_manager.py
=================
Module 1 — Configuration Manager
RCS-Radar-SDR  RM2026 Arena System

Reads config.json, resolves exact RF centre frequencies per
RM2026 Rulebook, and emits frozen FreqPlan and PhyConfig dataclasses
consumed by every downstream module.

RM2026 Official Frequency Plan
─────────────────────────────────────────────────────────────
  Channel            │ Freq (MHz) │ RF BW (MHz) │ Power   │ Sensitivity
  ───────────────────┼────────────┼─────────────┼─────────┼───────────
  Red broadcast       │ 433.20     │ 0.54        │ −60 dBm │ 1.5756
  Red jammer L1       │ 432.20     │ 0.94        │ −10 dBm │ 2.8323
  Red jammer L2       │ 432.50     │ 0.86        │ −10 dBm │ 2.5809
  Red jammer L3       │ 432.80     │ 0.25        │ −10 dBm │ 0.6646
  Blue broadcast      │ 433.92     │ 0.54        │ −60 dBm │ 1.5756
  Blue jammer L1      │ 434.92     │ 0.94        │ −10 dBm │ 2.8323
  Blue jammer L2      │ 434.62     │ 0.86        │ −10 dBm │ 2.5809
  Blue jammer L3      │ 434.32     │ 0.25        │ −10 dBm │ 0.6646

Sensitivity → deviation relationship
────────────────────────────────────
  sensitivity = 2 * pi * deviation / sample_rate
  deviation = sensitivity * sample_rate / (2 * pi)

Strategy: "listen to opponent"
  We are Red  → listen to Blue's broadcast + Blue's active jammer.
  We are Blue → listen to Red's broadcast  + Red's active jammer.

CLI smoke test:
  python config_manager.py [path/to/config.json]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── RM2026 Frequency tables (Hz) — official spec ────────────────────────────

_BROADCAST: dict[str, float] = {
    "blue": 433_920_000.0,
    "red":  433_200_000.0,
}

_JAMMERS: dict[str, dict[int, Optional[float]]] = {
    "blue": {
        0: None,
        1: 434_920_000.0,
        2: 434_620_000.0,
        3: 434_320_000.0,
    },
    "red": {
        0: None,
        1: 432_200_000.0,
        2: 432_500_000.0,
        3: 432_800_000.0,
    },
}

# Sensitivity and RF bandwidth per channel
_SENSITIVITY: dict[str, float] = {
    "broadcast": 1.5756,
    "jammer_L1": 2.8323,
    "jammer_L2": 2.5809,
    "jammer_L3": 0.6646,
}

_RF_BW_HZ: dict[str, float] = {
    "broadcast": 540_000.0,
    "jammer_L1": 940_000.0,
    "jammer_L2": 860_000.0,
    "jammer_L3": 250_000.0,
}

# AD9363 / PlutoSDR constraints
_PLUTO_MIN_SR_HZ = 521_000
_PLUTO_MAX_SR_HZ = 20_000_000
_SR_STEP_HZ      = 500_000
_GUARD_FACTOR    = 1.6
_DEFAULT_2GFSK_SR = 1_000_000   # default RX sample rate for 2-GFSK mode


# ─── Public dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhyConfig:
    """
    PHY-layer configuration for DSP processing mode selection.

    Attributes
    ----------
    mode : str
        "2gfsk" (default) or "4rrcfsk_legacy".
    sample_rate : float
        ADC sample rate in Hz.
    sps : int
        Samples per symbol.
    bt : float
        Gaussian bandwidth-time product (2-GFSK only).
    span : int
        Gaussian filter span in symbols (2-GFSK only).
    deviation_hz : float
        Peak frequency deviation in Hz.
    sub_block_syms : int
        Symbols between clock phase re-search.
    score_mode : str
        Clock recovery scoring function.
    access_code_mode : str
        "info", "jammer", or "both".
    sensitivity : float
        FM sensitivity for deviation computation.
    """
    mode: str = "2gfsk"
    sample_rate: float = 1_000_000.0
    sps: int = 52
    bt: float = 0.35
    span: int = 4
    deviation_hz: float = 250_000.0
    sub_block_syms: int = 512
    score_mode: str = "gfsk2_variance"
    access_code_mode: str = "both"
    sensitivity: float = 1.5756


@dataclass(frozen=True)
class FreqPlan:
    """
    Immutable frequency plan consumed by every module.

    Attributes
    ----------
    our_color           : "red" | "blue"  — our team colour
    opponent_color      : opposite of our_color
    jammer_level        : 0–3  (0 = no active jammer)
    broadcast_freq_hz   : centre frequency of the opponent's broadcast signal
    jammer_freq_hz      : centre frequency of the active jammer (None if level 0)
    center_freq_hz      : PlutoSDR LO frequency (midpoint of broadcast + jammer span)
    sample_rate_hz      : ADC/DAC clock; guaranteed to accommodate both signals
    broadcast_offset_hz : broadcast offset from PlutoSDR centre (signed Hz)
    jammer_offset_hz    : jammer offset from PlutoSDR centre (0.0 when absent / L3)
    channelize          : True → broadcast and jammer are in distinct sub-bands;
                          the DSP layer must digital-channelise before FM-demod
    """
    our_color           : str
    opponent_color      : str
    jammer_level        : int
    broadcast_freq_hz   : float
    jammer_freq_hz      : Optional[float]
    center_freq_hz      : float
    sample_rate_hz      : int
    broadcast_offset_hz : float
    jammer_offset_hz    : float
    channelize          : bool
    our_broadcast_freq_hz : float   # own team's TX broadcast frequency (Hz)
    broadcast_rf_bw_hz   : float   # RF bandwidth of broadcast signal
    jammer_rf_bw_hz      : float   # RF bandwidth of active jammer signal

    # ── helpers ──────────────────────────────────────────────────────────────

    def digital_lpf_cutoff_hz(self) -> float:
        """LPF cut-off for broadcast channel extraction (half BW + guard)."""
        return (self.broadcast_rf_bw_hz / 2) * 1.2

    def summary(self) -> str:
        lines = [
            "─" * 56,
            f"  FreqPlan  │ We are {self.our_color.upper()}, "
            f"opponent is {self.opponent_color.upper()}",
            "─" * 56,
            f"  Broadcast : {self.broadcast_freq_hz / 1e6:.3f} MHz"
            f"  (offset {self.broadcast_offset_hz / 1e3:+.1f} kHz)",
        ]
        if self.jammer_freq_hz is not None:
            lines.append(
                f"  Jammer L{self.jammer_level} : {self.jammer_freq_hz / 1e6:.3f} MHz"
                f"  (offset {self.jammer_offset_hz / 1e3:+.1f} kHz)"
            )
        else:
            lines.append("  Jammer    : None (Level 0 — no active jammer)")
        lines += [
            f"  SDR centre: {self.center_freq_hz / 1e6:.3f} MHz",
            f"  Own TX    : {self.our_broadcast_freq_hz / 1e6:.3f} MHz",
            f"  Sample rate: {self.sample_rate_hz / 1e6:.2f} MSPS",
            f"  Channelise : {self.channelize}",
            f"  LPF cutoff : {self.digital_lpf_cutoff_hz() / 1e3:.0f} kHz "
            f"(broadcast channel extraction)",
            "─" * 56,
        ]
        return "\n".join(lines)


# ─── ConfigManager ─────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Reads config.json and exposes a resolved :class:`FreqPlan`.

    Parameters
    ----------
    config_path : path to the JSON file; defaults to ``config.json`` next to this file.

    Usage::

        mgr = ConfigManager().load()
        plan = mgr.plan          # FreqPlan (frozen dataclass)
        uri  = mgr.pluto_uri     # "ip:192.168.2.1"
    """

    _DEFAULT_PATH = Path(__file__).parent / "config.json"

    def __init__(self, config_path: str | Path | None = None):
        self._path: Path  = Path(config_path) if config_path else self._DEFAULT_PATH
        self._raw:  dict  = {}
        self._plan: Optional[FreqPlan] = None
        self._phy:  PhyConfig = PhyConfig()

    # ── public ───────────────────────────────────────────────────────────────

    def load(self) -> "ConfigManager":
        """Parse config.json and resolve FreqPlan.  Returns self for chaining."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._path}\n"
                "Copy config.json from the repo root and adjust team_color / "
                "target_jammer_level."
            )
        with open(self._path, encoding="utf-8") as fh:
            raw = json.load(fh)
        # Strip JSON5-style comment keys silently
        self._raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        self._plan = self._resolve()
        return self

    @property
    def plan(self) -> FreqPlan:
        if self._plan is None:
            raise RuntimeError("Call .load() before accessing .plan")
        return self._plan

    @property
    def pluto_uri(self) -> str:
        """PlutoSDR URI for TX (and RX when a single board is used for both)."""
        return str(self._raw.get("pluto_uri", "ip:192.168.2.1"))

    @property
    def pluto_rx_uri(self) -> str:
        """PlutoSDR URI for the dedicated RX board (falls back to pluto_uri)."""
        return str(self._raw.get("pluto_rx_uri", self.pluto_uri))

    @property
    def rx_gain_db(self) -> int:
        """
        Manual RX gain in dB.

        Selection guidance
        ──────────────────
        The +10 dBm jammer is roughly 70 dB stronger than the −60 dBm broadcast.
        The AD9363 12-bit ADC has ~72 dB dynamic range.  To avoid ADC saturation
        when the jammer is present:

          recommended starting point: rx_gain_db = 20–30 dB
          then tune so that IQ RMS ≈ 0.05–0.30 (use --capture / --diagnose).
        """
        return int(self._raw.get("rx_gain_db", 30))

    @property
    def tx_attenuation_db(self) -> int:
        """Pluto TX attenuation 0..89 dB  (0 = max power)."""
        return int(self._raw.get("tx_attenuation_db", 0))

    @property
    def rx_buf_size(self) -> int:
        """IQ samples per callback block (must be power-of-2)."""
        v = int(self._raw.get("rx_buf_size", 1 << 18))
        if v & (v - 1):
            raise ValueError(f"rx_buf_size must be a power of 2, got {v}")
        return v

    @property
    def phy_config(self) -> PhyConfig:
        """Resolved PHY-layer configuration."""
        return self._phy

    # ── private ──────────────────────────────────────────────────────────────

    def _resolve(self) -> FreqPlan:
        our_color = str(self._raw.get("team_color", "red")).lower().strip()
        if our_color not in ("red", "blue"):
            raise ValueError(
                f"team_color must be 'red' or 'blue', got {our_color!r}"
            )
        level = int(self._raw.get("target_jammer_level", 0))
        if level not in (0, 1, 2, 3):
            raise ValueError(
                f"target_jammer_level must be 0..3, got {level}"
            )

        phy_mode = str(self._raw.get("phy_mode", "2gfsk")).lower().strip()

        opponent      = "blue" if our_color == "red" else "red"
        broadcast_hz  = _BROADCAST[opponent]
        jammer_hz     = _JAMMERS[opponent][level]

        # Perc-channel RF bandwidths
        bc_rf_bw  = _RF_BW_HZ["broadcast"]
        jam_rf_bw = (_RF_BW_HZ.get(f"jammer_L{level}", bc_rf_bw)
                     if jammer_hz is not None else 0.0)

        # ── Centre frequency and sample rate ────────────────────────────────
        if jammer_hz is None:
            # Level 0: single channel, broadcast only.
            center_hz  = broadcast_hz
            sr_hz      = _DEFAULT_2GFSK_SR if phy_mode == "2gfsk" else 2_500_000
            bc_offset  = 0.0
            jam_offset = 0.0
            channelize = False
        else:
            # Levels 1-3: two distinct channels (L3 is no longer == broadcast).
            separation = abs(jammer_hz - broadcast_hz)
            center_hz  = (broadcast_hz + jammer_hz) / 2
            bc_offset  = broadcast_hz - center_hz
            jam_offset = jammer_hz    - center_hz

            edge_hz = abs(bc_offset) + bc_rf_bw / 2
            min_sr  = _next_pluto_sr(edge_hz * 2 * _GUARD_FACTOR)
            sr_hz   = max(min_sr, 3_000_000)
            channelize = True

        # ── Build PhyConfig ─────────────────────────────────────────────────
        if phy_mode == "2gfsk":
            # Compute deviation from broadcast sensitivity
            sensitivity = _SENSITIVITY["broadcast"]
            deviation = sensitivity * _DEFAULT_2GFSK_SR / (2.0 * math.pi)
            self._phy = PhyConfig(
                mode="2gfsk",
                sample_rate=_DEFAULT_2GFSK_SR,
                sps=52,
                bt=0.35,
                span=4,
                deviation_hz=round(deviation, 1),
                sub_block_syms=512,
                score_mode="gfsk2_variance",
                access_code_mode="both",
                sensitivity=sensitivity,
            )
        else:
            self._phy = PhyConfig(
                mode="4rrcfsk_legacy",
                sample_rate=float(sr_hz),
                sps=sr_hz // 250_000,
                deviation_hz=250_000.0,
                score_mode="fsk4_energy",
            )

        return FreqPlan(
            our_color             = our_color,
            opponent_color        = opponent,
            jammer_level          = level,
            broadcast_freq_hz     = broadcast_hz,
            jammer_freq_hz        = jammer_hz,
            center_freq_hz        = center_hz,
            sample_rate_hz        = sr_hz,
            broadcast_offset_hz   = bc_offset,
            jammer_offset_hz      = jam_offset,
            channelize            = channelize,
            our_broadcast_freq_hz = _BROADCAST[our_color],
            broadcast_rf_bw_hz    = bc_rf_bw,
            jammer_rf_bw_hz       = jam_rf_bw,
        )


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _next_pluto_sr(required_hz: float) -> int:
    """
    Round *required_hz* up to the nearest multiple of _SR_STEP_HZ,
    then clamp to [_PLUTO_MIN_SR_HZ, _PLUTO_MAX_SR_HZ].
    """
    sr = math.ceil(required_hz / _SR_STEP_HZ) * _SR_STEP_HZ
    return int(min(max(sr, _PLUTO_MIN_SR_HZ), _PLUTO_MAX_SR_HZ))


# ─── Module-level one-liner ────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> ConfigManager:
    """Load config.json and return a ready :class:`ConfigManager`."""
    return ConfigManager(path).load()


# ─── PlutoSDR auto-detection ───────────────────────────────────────────────────

def detect_pluto_rx_uri() -> "str | None":
    """
    Probe for a PlutoSDR receiver (USB or network).

    Strategy (in order)
    ───────────────────
    1. libiio ``iio.scan_contexts()`` — returns both USB and IP contexts
    2. Direct adi.Pluto probe on USB URIs (``usb:`` / ``usb:1.0`` etc.)
    3. Direct adi.Pluto probe on IP candidates and mDNS hostname

    Each candidate is verified by actually opening an IIO context so we
    never return a URI that would later fail with "No device found".

    Returns the first working URI or ``None``.
    """
    # ─ helper: try to open adi.Pluto and return True on success ─────────────
    def _probe(uri: str) -> bool:
        try:
            import adi  # type: ignore
            sdr = adi.Pluto(uri=uri)
            try:
                sdr.rx_destroy_buffer()
            except Exception:
                pass
            return True
        except Exception:
            return False

    # Method 1: libiio context scan (covers USB + IP simultaneously)
    try:
        import iio  # type: ignore
        for uri, desc in iio.scan_contexts().items():
            if any(kw in desc for kw in ("PlutoSDR", "ADALM", "AD9361", "AD9363",
                                         "Analog Devices")):
                if _probe(uri):
                    return uri
    except Exception:
        pass

    # Method 2: USB direct (Windows: Pluto shows up as usb: context when
    # the WinUSB/libusbK driver is active, even without RNDIS networking)
    for uri in ("usb:", "usb:1.0", "usb:2.0", "usb:3.0",
                "usb:1.1", "usb:2.1", "usb:3.1"):
        if _probe(uri):
            return uri

    # Method 3: IP network scan (RNDIS / Ethernet adapter)
    import socket
    for ip in ("192.168.2.1", "192.168.3.1", "192.168.1.1", "pluto.local"):
        try:
            with socket.create_connection((ip, 30431), timeout=1.0):
                pass
        except OSError:
            continue
        uri = f"ip:{ip}"
        if _probe(uri):
            return uri

    return None


def _pluto_diagnose_hint() -> str:
    """Return a multi-line Windows troubleshooting hint for 'No device found'."""
    return (
        "\n"
        "  ╔══ PlutoSDR NOT FOUND ═ Troubleshooting ═══════════════════════════════╗\n"
        "  ║ A) IIO Oscilloscope (libiio) installed?                               ║\n"
        "  ║    https://github.com/analogdevicesinc/iio-oscilloscope/releases       ║\n"
        "  ║ B) USB RNDIS driver active?  In Device Manager look for:              ║\n"
        "  ║    'RNDIS/Ethernet Gadget'  (should be under Network Adapters)        ║\n"
        "  ║    If shown as Unknown Device → install via Zadig (RNDIS / WinUSB)    ║\n"
        "  ║ C) Can you ping the Pluto?  Run:  ping 192.168.2.1                    ║\n"
        "  ║    Yes → set pluto_rx_uri = \"ip:192.168.2.1\" in config.json           ║\n"
        "  ║    No  → try USB URI:  set pluto_rx_uri = \"usb:\" in config.json       ║\n"
        "  ║ D) Verify libiio sees the device:                                     ║\n"
        "  ║    python -c \"import iio; print(iio.scan_contexts())\"                 ║\n"
        "  ╚═══════════════════════════════════════════════════════════════════════╝"
    )


def save_rx_uri_to_config(config_path: "str | Path | None", uri: str) -> bool:
    """
    Write *uri* as ``pluto_rx_uri`` into *config_path* (in-place JSON update).
    Preserves all existing keys including ``_comment``.
    Returns ``True`` on success.
    """
    path = Path(config_path) if config_path else Path(__file__).parent / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["pluto_rx_uri"] = uri
        path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        print(f"[WARN] Could not save detected RX URI to config: {exc}")
        return False


# ─── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    _mgr  = load_config(_path)
    print(_mgr.plan.summary())
    print(f"  Pluto URI   : {_mgr.pluto_uri}")
    print(f"  RX Gain     : {_mgr.rx_gain_db} dB")
    print(f"  TX Atten    : {_mgr.tx_attenuation_db} dB")
    print(f"  RX buf size : {_mgr.rx_buf_size} samples")

    print("\n[AUTO] Scanning for PlutoSDR …", end=" ", flush=True)
    _found = detect_pluto_rx_uri()
    if _found:
        print(f"found: {_found}")
    else:
        print("not found")
        print(_pluto_diagnose_hint())
