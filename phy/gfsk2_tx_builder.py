"""
phy/gfsk2_tx_builder.py
========================
Reusable 2-GFSK TX builder — shared by tx_gfsk2_test.py and main.py.

Builds a complete 2-GFSK cyclic TX buffer from RM frame bytes:

  RM frame bytes
  → split into 15-byte payload chunks
  → Access Code (64b) + Header 0x000F000F (32b) + Payload (120b)
  → MSB-first bits
  → gfsk2_modulate_bits()
  → tile for cyclic DMA

Usage::

    from phy.gfsk2_tx_builder import build_gfsk2_tx_iq
    iq = build_gfsk2_tx_iq(frame, mode="info", repeats=4)
    # iq is a complex64 array ready for Pluto sdr.tx()
"""

from __future__ import annotations

import struct
import numpy as np

from phy.gfsk2_modem import gfsk2_modulate_bits
from phy.air_packet import ac_to_bits

# ── PHY constants ────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE  = 1_000_000
DEFAULT_SPS          = 52
DEFAULT_BT           = 0.35
DEFAULT_SPAN         = 4

# Access Codes (64-bit integers)
AC_INFO   = 0x2F6F4C74B914492E
AC_JAMMER = 0x16E8D377151C712D

# Pluto DAC full-scale reference (−6 dBFS on 16-bit DMA)
DAC_FULL_SCALE = float(2**14 * 0.5)

# Air-packet constants
PAYLOAD_BYTES = 15
HEADER_U16    = 15


def build_gfsk2_tx_iq(
    frame: bytes,
    *,
    mode: str = "info",
    deviation_hz: float = 250_000.0,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    sps: int = DEFAULT_SPS,
    bt: float = DEFAULT_BT,
    span: int = DEFAULT_SPAN,
    repeats: int = 4,
) -> np.ndarray:
    """
    Build a cyclic 2-GFSK IQ buffer from RM frame bytes.

    Parameters
    ----------
    frame : bytes
        Complete RM frame (SOF … CRC16).
    mode : str
        "info" (AC=0x2F…) or "jammer" (AC=0x16…).
    deviation_hz : float
        Peak frequency deviation in Hz.
    sample_rate : float
        DAC sample rate in Hz.
    sps : int
        Samples per symbol.
    bt : float
        Gaussian bandwidth-time product.
    span : int
        Gaussian filter span in symbols.
    repeats : int
        Number of frame repetitions for glitch-free cyclic DMA.

    Returns
    -------
    np.ndarray
        Complex64 IQ samples scaled to Pluto DAC range.
    """
    ac_int = AC_INFO if mode == "info" else AC_JAMMER

    # Build all air-packet bits
    all_bits: list[int] = []
    for offset in range(0, len(frame), PAYLOAD_BYTES):
        chunk = frame[offset:offset + PAYLOAD_BYTES]
        if len(chunk) < PAYLOAD_BYTES:
            chunk = chunk + b"\x00" * (PAYLOAD_BYTES - len(chunk))
        all_bits.extend(_build_air_packet_bits(chunk, ac_int))

    # Modulate
    iq_frame = gfsk2_modulate_bits(
        all_bits,
        sps=sps,
        bt=bt,
        span=span,
        deviation_hz=deviation_hz,
        sample_rate=sample_rate,
    )

    # Tile and scale to DAC
    iq_tx = np.tile(iq_frame, repeats).astype(np.complex64)
    iq_tx = (iq_tx * DAC_FULL_SCALE).astype(np.complex64)
    return iq_tx


def build_air_packet_bits_from_payload(payload: bytes, ac_int: int) -> list[int]:
    """
    Build air-packet bit sequence for a single 15-byte payload.

    Returns list of 216 bits (64 AC + 32 Header + 120 Payload).
    """
    assert len(payload) == PAYLOAD_BYTES
    return _build_air_packet_bits(payload, ac_int)


# ── helpers ──────────────────────────────────────────────────────────────────

def _bytes_to_bits_msb(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits: list[int] = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _build_air_packet_bits(payload: bytes, ac_int: int) -> list[int]:
    """Build air-packet: AC(64b) + Header(32b) + Payload(120b)."""
    bits = ac_to_bits(ac_int)
    header_bytes = struct.pack(">HH", HEADER_U16, HEADER_U16)
    bits.extend(_bytes_to_bits_msb(header_bytes))
    bits.extend(_bytes_to_bits_msb(payload))
    return bits
