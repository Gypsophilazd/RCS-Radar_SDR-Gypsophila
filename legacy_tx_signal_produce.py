"""
tx_signal_produce.py
====================
Module 2 — TX Signal Producer  (PlutoSDR / pyadi-iio)
RCS-Radar-SDR  RM2026 Arena System

Generates the 4-RRC-FSK protocol frame (0xA5/CRC8/CRC16) and drives
ADALM-PLUTO in cyclic-TX mode for continuous broadcast.

Arena Simulation mode
─────────────────────
When *simulate_arena=True*, the module digitally mixes two signals
into a single IQ stream before handing it to the Pluto DAC:

  Signal A — "Broadcast"   at  broadcast_offset_hz from the LO
             target amplitude  ← −60 dBm (very weak, information-bearing)

  Signal B — "Jammer tone" at  jammer_offset_hz from the LO
             target amplitude  ← +10 dBm equivalent (strong, to stress
             the receiver's dynamic range)

This replicates the real arena RF environment on a single bench setup.

Usage::

    from config_manager      import load_config
    from tx_signal_produce   import PlutoTxProducer

    mgr  = load_config()
    tx   = PlutoTxProducer(mgr, key="RM2026", simulate_arena=False)
    tx.start()          # non-blocking; Pluto runs in cyclic mode
    ...
    tx.stop()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config_manager import ConfigManager

try:
    import adi
    _PLUTO_AVAILABLE = True
except ImportError:
    _PLUTO_AVAILABLE = False

# ── DSP constants (must match RX chain) ──────────────────────────────────────
_BAUD_RATE     = 250_000
_FSK_DEV_HZ    = 250_000
_RRC_ALPHA     = 0.25
_RRC_SPAN      = 11            # → span × SPS taps  (e.g. 11×10=110 @ 2.5 MSPS)
# SPS is computed dynamically: sps = plan.sample_rate_hz // _BAUD_RATE
# (2.5 MSPS → SPS=10;  2.0 MSPS → SPS=8)

# AD9363 uses a 12-bit DAC, but Pluto DMA buffer is 16-bit.
# pyadi-iio maps complex64 directly into the 16-bit DMA; the practical
# full-scale reference is 2^14 (=16384).  Using 2^14 * 0.5 = 8192 gives
# −6 dBFS headroom and matches the proven rx_4fsk_pipeline.py reference.
# The old value of 2047 only utilised 12.5% of the dynamic range (−18 dBFS),
# making transmitted power 4× (12 dB) too low.
DAC_FULL_SCALE        = int(2**14 * 0.5)   # 8192
DIGITAL_POWER_DIFF_DB = 35.0
INFO_TO_JAMMER_AMP    = 10 ** (-DIGITAL_POWER_DIFF_DB / 20.0)


class PlutoTxProducer:
    """
    Builds and transmits the RM2026 key frame via PlutoSDR.

    Parameters
    ----------
    config         : loaded ConfigManager
    key            : 6-char ASCII jammer key  (e.g. "RM2026")
    simulate_arena : mix broadcast + jammer tone to simulate full arena RF
    """

    def __init__(
        self,
        config: "ConfigManager",
        key: str = "RM2026",
        simulate_arena: bool = False,
        test_tx_enabled: bool = False,
    ):
        self._cfg            = config
        self._key            = key
        self._simulate_arena = simulate_arena
        self._test_tx_enabled = test_tx_enabled
        self._sdr            = None

    # ── public ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open Pluto, build IQ, and begin cyclic transmission."""
        if not self._test_tx_enabled:
            raise RuntimeError(
                "TX is disabled by default. "
                "Use --test-tx-enable only in legal test conditions."
            )
        if not _PLUTO_AVAILABLE:
            raise RuntimeError("pyadi-iio not installed.  Run: pip install pyadi-iio")
        plan = self._cfg.plan
        sr   = plan.sample_rate_hz
        sps  = sr // _BAUD_RATE

        info_iq = self._build_iq(self._key, sr=sr, sps=sps)
        # TX LO is already set to our_broadcast_freq_hz (own team's channel).
        # No additional frequency offset is needed — the signal sits at DC
        # relative to the TX LO.  broadcast_offset_hz is an RX-side concept
        # (offset of the OPPONENT's signal inside the SDR capture band).

        if self._simulate_arena:
            tx_iq = self._mix_arena_signals(info_iq, sr=sr)
        else:
            tx_iq = self._scale_to_dac(info_iq)

        self._sdr = self._open_pluto(sr=sr)
        self._sdr.tx_cyclic_buffer = True
        self._sdr.tx(tx_iq)
        print(f"[PLUTO-TX] Cyclic TX running  ({len(tx_iq)} samples)")

    def stop(self) -> None:
        """Halt TX and release the Pluto handle."""
        if self._sdr is not None:
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None

    # ── private ──────────────────────────────────────────────────────────────

    def _open_pluto(self, sr: int):
        plan = self._cfg.plan
        print(f"[PLUTO-TX] Connecting to {self._cfg.pluto_uri} ...")
        sdr  = adi.Pluto(uri=self._cfg.pluto_uri)
        # TX LO must be OWN team's broadcast frequency, not the opponent's.
        # plan.center_freq_hz = opponent's frequency (used as RX SDR centre).
        # plan.our_broadcast_freq_hz = own team's TX channel.
        sdr.tx_lo                   = int(plan.our_broadcast_freq_hz)
        sdr.sample_rate             = sr
        sdr.tx_rf_bandwidth         = sr
        sdr.tx_hardwaregain_chan0   = -self._cfg.tx_attenuation_db
        print(f"[PLUTO-TX] {plan.our_broadcast_freq_hz/1e6:.3f} MHz  "
              f"SR={sr/1e6:.2f} MSPS  atten={self._cfg.tx_attenuation_db} dB")
        return sdr

    def _build_iq(self, key: str, sr: int, sps: int) -> np.ndarray:
        """
                Build one complete cyclic information-wave buffer in float domain.

        Signal chain
        ────────────
          key bytes → frame bytes → bits → symbols →
          upsample(sps) → RRC filter → freq deviation scale →
                    cumsum (FM integration) → complex exponential in [-1, 1]
        """
        # Import frame builder from the existing digital twin
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from fsk_digital_twin import build_protocol_frame  # type: ignore

        frame_bytes = build_protocol_frame(key)
        bits        = _bytes_to_bits(bytes(frame_bytes))
        symbols     = _bits_to_symbols(bits)                 # {-3,-1,+1,+3}
        symbols_rep = np.tile(symbols, 4)   # 4× repeat for glitch-free cyclic DMA loop
                                            # (matches rx_pluto_pipeline.py PlutoTX)

        # RRC pulse shaping (same filter as RX matched filter)
        rrc     = _make_rrc(_RRC_ALPHA, _RRC_SPAN, sps)
        upsampl = np.zeros(len(symbols_rep) * sps)
        upsampl[::sps] = symbols_rep
        freq_pulse = np.convolve(upsampl, rrc, mode="full")[len(rrc) // 2:]
        freq_pulse = freq_pulse[:len(upsampl)]

        # Scale to Hz and FM modulate
        freq_hz = freq_pulse * (_FSK_DEV_HZ / 3.0)
        phase   = np.cumsum(freq_hz) * 2 * np.pi / sr
        return np.exp(1j * phase).astype(np.complex64)

    def _mix_arena_signals(self, info_iq: np.ndarray, sr: int) -> np.ndarray:
        """
        Mix weak information wave and strong jammer wave in float domain.

        In real arena operation the jammer is a separate transmitter;
        this method is only used in bench simulation mode.

        DSP rules enforced here:
          1. Digital power difference limited to 35 dB.
          2. All intermediate processing stays in [-1, 1].
          3. No per-sample clipping/normalization.
          4. One global static scale is applied to the entire buffer.
        """
        plan = self._cfg.plan
        jammer_iq = self._build_jammer_wave(len(info_iq), plan.jammer_offset_hz, sr)
        mixed_iq = jammer_iq + (info_iq * INFO_TO_JAMMER_AMP)

        global_max = float(np.max(np.abs(mixed_iq)))
        if global_max > 1.0:
            mixed_iq = mixed_iq / global_max

        return self._scale_to_dac(mixed_iq)

    @staticmethod
    def _apply_frequency_offset(iq: np.ndarray, offset_hz: float, sr: int) -> np.ndarray:
        """Shift a complex buffer to its assigned RF offset relative to the LO."""
        if offset_hz == 0.0:
            return iq.astype(np.complex64)
        n = len(iq)
        t = np.arange(n, dtype=np.float32) / sr
        rotator = np.exp(1j * 2 * np.pi * offset_hz * t).astype(np.complex64)
        return (iq * rotator).astype(np.complex64)

    @staticmethod
    def _build_jammer_wave(num_samples: int, offset_hz: float, sr: int) -> np.ndarray:
        """Create a full-scale constant-envelope jammer tone in float domain."""
        if offset_hz == 0.0:
            return np.ones(num_samples, dtype=np.complex64)
        t = np.arange(num_samples, dtype=np.float32) / sr
        return np.exp(1j * 2 * np.pi * offset_hz * t).astype(np.complex64)

    @staticmethod
    def _scale_to_dac(iq: np.ndarray) -> np.ndarray:
        """Apply one final global scale into Pluto DAC counts."""
        return (iq.astype(np.complex64) * DAC_FULL_SCALE).astype(np.complex64)


# ─── Helper functions ──────────────────────────────────────────────────────────

def _bytes_to_bits(data: bytes) -> list[int]:
    """MSB-first byte-to-bit expansion."""
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]


def _bits_to_symbols(bits: list[int]) -> np.ndarray:
    """Dibit → {-3,-1,+1,+3}  (MSB first)."""
    table = {0b00: -3, 0b01: -1, 0b10: +1, 0b11: +3}
    syms  = [table[(bits[i] << 1) | bits[i + 1]] for i in range(0, len(bits) - 1, 2)]
    return np.array(syms, dtype=np.float32)


def _make_rrc(alpha: float, span: int, sps: int) -> np.ndarray:
    """Root Raised Cosine FIR  —  span*sps taps (even length)."""
    n_taps = span * sps
    t      = (np.arange(n_taps) - (n_taps - 1) / 2) / sps   # exact fractional centre
    h      = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + alpha * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-6:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num   = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            denom = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i]  = num / denom
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float32)


# ─── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).parent))
    from config_manager import load_config

    mgr = load_config()
    print(mgr.plan.summary())
    print("Starting PlutoTxProducer for 10 seconds…")
    tx = PlutoTxProducer(mgr, key="RM2026", simulate_arena=False)
    tx.start()
    time.sleep(10)
    tx.stop()
    print("TX stopped.")
