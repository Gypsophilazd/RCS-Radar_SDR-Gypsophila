#!/usr/bin/env python3
"""
4-RRC-FSK Split TX/RX Pipeline  —  RoboMaster 2026 Radar EW
=========================================================
TX Hardware : ADALM-PLUTO  (pyadi-iio)
RX Hardware : RTL-SDR V4   (pyrtlsdr + local librtlsdr.dll)
Modulation  : 4-RRC-FSK, NO PREAMBLE

Physical-Layer Math
-------------------
  Δf  = Symbol Rate = 250 kHz
  SPS = SR / Baud  = 2e6 / 250e3  = 8
  BW  = 2*(Δf + 0.02 MHz) = 0.54 MHz
  f(s) = s * (Δf/3),  s ∈ {-3,-1,+1,+3}

DSP Chain
---------
  RTL-SDR → FLL → FM-demod → RRC-MF → AGC →
  M&M TED → Slicer → SOF-hunt → CRC-check → parse

Dependencies
------------
  pip install numpy scipy pyadi-iio pyrtlsdr
  DLL : librtlsdr.dll + libusb-1.0.dll must be in the same folder as this script.
"""

import os
import sys
import struct
import threading
import queue

# ── DLL bootstrap ─────────────────────────────────────────────────────────────
# pyrtlsdr uses ctypes.util.find_library() which on Windows searches os.environ
# ['PATH'].  os.add_dll_directory() only covers .pyd extensions, NOT ctypes DLLs.
# Both must be applied BEFORE any 'from rtlsdr import ...' statement.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. Traditional PATH approach (works with ctypes.util.find_library on Windows)
os.environ['PATH'] = _SCRIPT_DIR + os.pathsep + os.environ.get('PATH', '')
# 2. Python 3.8+ DLL directory (belt-and-suspenders for .pyd modules)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(_SCRIPT_DIR)

import numpy as np
from scipy.signal import lfilter, firwin, find_peaks
from collections import deque   # kept for potential future use

# ─────────────────────────── Config ────────────────────────────
SAMPLE_RATE     = 2_000_000    # 2 MSPS
BAUD_RATE       = 250_000      # 250 kBaud
SPS             = SAMPLE_RATE // BAUD_RATE   # = 8
CENTER_FREQ     = 433_200_000  # 433.2 MHz
GAIN            = 5            # RTL-SDR gain (dB); strong antenna — keep low to avoid ADC saturation
BUF_SAMPLES     = 256 * 1024   # ~128 ms per callback burst

# ── Pluto TX config ──────────────────────────────────────────────────────────
PLUTO_URI       = "ip:192.168.2.1"   # change to "ip:pluto.local" if needed
# TX Power target: −60 dBm (per RoboMaster rules)
# Pluto hardware gain range: −0 dB to −89.75 dB attenuation
TX_ATTENUATION  = 0            # dB → max Pluto power (~+7 dBm); -10 still gave SNR≈-5dB at bench
TX_CYCLIC       = True         # broadcast frame continuously

FSK_DEVIATION   = 250_000      # Δf = 250 kHz  (= Symbol Rate per spec)
RRC_ALPHA       = 0.25
RRC_SPAN        = 11           # → 11*8 = 88 taps (per spec)

# Müller-Mueller TED loop filter gains
# RTL-SDR crystal accuracy ~20 ppm → drift ≤ 20e-6 * SPS ≈ 1.6e-4 samples/symbol
# Required Kp: correction/symbol ≥ drift → Kp * |e_unit| ≥ 1.6e-4
# |e_unit| ≈ 0.7 (from measurement) → Kp ≥ 2e-4.  Use 0.002 for margin.
# Keep Ki very small (0 for open-loop, tiny for hardware clock drift only).
GARDNER_Kp      = 0.002        # Was 0.04 — reduced for stability
GARDNER_Ki      = 0.000002     # Was 0.001 — reduced to near-zero

# AFC: deliberately DISABLED.
# FM demod mean = carrier_offset_norm + symbol_mean_norm.
# For RM2026 frame: symbol_mean = -1.067, which is 31% of one decision interval.
# Any attempt to subtract FM mean removes DATA, not just carrier offset.
# Slicer tolerance: half-interval = Δf/6 = 83333/2 ≈ 41.7 kHz.
# RTL-SDR crystal offset ≈ 4-5 kHz << 41.7 kHz → zero AFC needed.
AFC_ALPHA       = 0.0   # kept as placeholder; not used in FMDemod

# Protocol constants
SOF             = 0xA5
CMDS_LEN = {    # CmdID → payload_bytes (excluding CmdID itself)
    0x0A01: 24,   # Enemy Coordinates
    0x0A02: 12,   # Enemy HP
    0x0A03: 10,   # Enemy Ammo
    0x0A04:  8,   # Macro Status
    0x0A05: 36,   # Enemy Buffs
    0x0A06:  6,   # Interference Key (ASCII)
}
MIN_FRAME_LEN   = 5 + 2 + 0 + 2   # header(4)+crc8(1)+cmdid(2)+0+crc16(2) = 9

# ═══════════════════════════════════════════════════════════════
# 1. RRC Filter (88 taps, even length)
# ═══════════════════════════════════════════════════════════════
def rrc_filter(alpha: float, span: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine FIR  —  88 taps (span*sps, even length per spec).

    Time-domain formula:
        h(t) = [ sin(π·t·(1-α)) + 4α·t·cos(π·t·(1+α)) ]
               ─────────────────────────────────────────
               π·t·[1 − (4α·t)²]
    where t is normalised to symbol period.
    """
    n_taps = span * sps                          # = 88
    t = (np.arange(n_taps) - (n_taps - 1) / 2) / sps
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1.0 + alpha * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-6:
            c = alpha / np.sqrt(2)
            h[i] = c * ((1 + 2/np.pi) * np.sin(np.pi / (4*alpha)) +
                        (1 - 2/np.pi) * np.cos(np.pi / (4*alpha)))
        else:
            num = (np.sin(np.pi * ti * (1 - alpha)) +
                   4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha)))
            den = np.pi * ti * (1 - (4 * alpha * ti)**2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h**2))
    return h

RRC_KERNEL = rrc_filter(RRC_ALPHA, RRC_SPAN, SPS)


# ═══════════════════════════════════════════════════════════════
# 2. FM Demodulator  (pure instantaneous frequency extraction)
# ═══════════════════════════════════════════════════════════════
class FMDemod:
    """
    Computes instantaneous frequency via conjugate-delay method.
      d_phase(n) = angle[ x(n) · conj(x(n-1)) ]   (radians/sample)
      f_hz(n)    = d_phase(n) · SR / (2π)

    Normalised output: f_norm = f_hz / (Δf/3)
    so ideal symbol levels map to {-3, -1, +1, +3}.

    NO AFC / DC correction:
      Any carrier offset f_off between TX and RX oscillators shifts all
      symbols by f_off / (Δf/3).  The slicer thresholds are at -2, 0, +2.
      Decision-region half-width = 1 unit = Δf/3 = 83.33 kHz.
      Errors only occur when |shift| > half-width = 41.67 kHz.
      RTL-SDR Blog V4 crystal tolerance ≈ ±20 ppm at 433 MHz → ±8.7 kHz.
      Pluto crystal ≈ ±25 ppm → ±10.8 kHz.
      Worst case combined: ±19.5 kHz << 41.67 kHz → no AFC required.

      Attempting to remove carrier offset by subtracting FM demod mean fails
      because FM mean = carrier_shift + symbol_mean×(Δf/3), and symbol_mean
      is data-dependent (RM2026 frame: symbol_mean = -1.067, shifting decision
      levels by ~1 unit and causing ~30% symbol errors).
    """
    def __init__(self, afc_alpha: float = AFC_ALPHA):   # afc_alpha unused
        self._prev = complex(1, 0)

    def demod(self, iq: np.ndarray) -> np.ndarray:
        iq_prev = np.empty_like(iq)
        iq_prev[0] = self._prev
        iq_prev[1:] = iq[:-1]
        self._prev = iq[-1]
        product = iq * np.conj(iq_prev)
        d_phase = np.angle(product)
        f_hz = d_phase * (SAMPLE_RATE / (2 * np.pi))
        return (f_hz / (FSK_DEVIATION / 3.0)).astype(np.float32)

    @property
    def freq_offset_hz(self) -> float:
        """Not estimated (AFC disabled); returns 0."""
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 3. Matched Filter  (RRC MF, stateful)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 2b.  IQ Pre-filter  (LPF before FM demod, reduces OOB noise ~5 dB)
# ═══════════════════════════════════════════════════════════════
_LPF_CUTOFF_HZ  = 300_000     # keep FSK signal (±250 kHz) + margin
_LPF_TAPS       = 63          # odd → linear-phase, group delay = 31 samples
_LPF_KERNEL     = firwin(_LPF_TAPS, cutoff=_LPF_CUTOFF_HZ,
                          fs=SAMPLE_RATE).astype(np.float32)

class PreFilter:
    """
    63-tap lowpass FIR applied to complex IQ before FM demod.
    Passband: DC ± 300 kHz,  SR = 2 MSPS.
    Reject out-of-band RTL-SDR noise to improve FM discriminator SNR by ~5 dB.
    Group delay = (63-1)//2 = 31 samples.  Since it is applied before FM demod
    and we are NOT decimating, no timing correction is needed.
    """
    def __init__(self):
        # Separate I and Q state (FIR evaluated separately for speed)
        self._zi = np.zeros(len(_LPF_KERNEL) - 1, dtype=np.complex64)

    def run(self, iq: np.ndarray) -> np.ndarray:
        out, self._zi = lfilter(_LPF_KERNEL, 1.0,
                                iq.astype(np.complex64), zi=self._zi)
        return out.astype(np.complex64)


class MatchedFilter:
    """Stateful RRC matched filter using overlap-save."""
    def __init__(self, kernel: np.ndarray):
        self._k = kernel
        self._state = np.zeros(len(kernel) - 1)

    def filter(self, x: np.ndarray) -> np.ndarray:
        out, self._state = lfilter(self._k, [1.0],
                                   x, zi=self._state * self._k[0])
        # lfilter does not preserve state correctly for FIR — do it manually:
        return out

    def run(self, x: np.ndarray) -> np.ndarray:
        """Correct stateful FIR using np.convolve overlap-save."""
        padded = np.concatenate([self._state, x])
        out = np.convolve(padded, self._k, mode='valid')
        self._state = padded[-(len(self._k)-1):]
        return out


# MF causal group delay = (N_taps - 1)//2  (88-tap: 43 samples = 5.375 symbols)
MF_GROUP_DELAY  = (RRC_SPAN * SPS - 1) // 2   # = 43
# Pre-filter group delay: (63-1)//2 = 31 samples
LPF_GROUP_DELAY = (_LPF_TAPS - 1) // 2        # = 31
# Transient info (kept for reference, NOT used as TED skip):
LPF_STARTUP     = _LPF_TAPS - 1               # = 62
MF_STARTUP      = RRC_SPAN * SPS - 1          # = 87
# TED initial skip = GROUP DELAY sum (31+43=74), NOT startup transient (149).
# Reason: group delay puts the first TED sample at frame symbol 0 (byte-aligned).
# Skip=149 overshoots to frame symbol 10 (10%4=2, byte-misaligned → SOF split).
TOTAL_CHAIN_DELAY = LPF_GROUP_DELAY + MF_GROUP_DELAY  # = 74  (group delay sum)

# ═══════════════════════════════════════════════════════════════
# 5. Symbol Sampler  (replaces M&M / Gardner TED)
# ═══════════════════════════════════════════════════════════════
_BUF_SIZE_MIN = 16 * SPS    # min samples after skip to trust phase search

class GardnerTED:
    """
    Adaptive blind-phase symbol sampler.  Named GardnerTED for drop-in
    compatibility.

    WHY NOT M&M / GARDNER TED:
      The M&M TED uses hard-decision feedback.  For a FIXED repeating frame
      (cyclic TX on Pluto) the symbol stream is NOT random; periodic patterns
      bias the M&M error term, driving omega to the lower clip limit (0.8×SPS)
      within ~100 symbols and never recovering.

    ALGORITHM:
      • Skip the first `initial_offset` samples (filter startup transient;
        default = TOTAL_CHAIN_DELAY = 149 = PreFilter + MF startup transients).
      • After skipping, run max-|value| phase search on the remainder:
        evaluate all SPS timing phases, pick the one with the highest mean |value|
        (the phase where the RRC pulse peaks correspond to symbol centers).
      • On subsequent large blocks (hardware streaming), re-run the phase search
        to adapt to any slow crystal drift.

    The skip correctly removes the FM-demod noise spikes caused by the
    PreFilter and MF starting from zero-state.  The phase search is then
    unambiguous since it sees only the valid signal region.
    """

    def __init__(self, sps: int = SPS, kp=None, ki=None,
                 initial_offset: int = 0):
        self._sps       = sps
        self._skip_left = initial_offset     # samples left to skip
        self._phase     = 0                  # best phase found on first search
        self._not_first = False              # True after first process() call
        self._omega     = float(sps)
        # Cross-chunk continuity: after processing a block we track how many
        # samples into the next block the first sample-point falls.
        # _next_offset=0 means the very first sample of the next block is a
        # sample point; _next_offset=k means skip k samples first.
        self._next_offset = 0

    def process(self, samples: np.ndarray) -> list:
        arr = np.asarray(samples, dtype=np.float32)
        n   = len(arr)

        # Consume skip budget (initial filter-startup transient)
        skip = min(self._skip_left, n)
        self._skip_left = max(0, self._skip_left - n)
        valid = arr[skip:]

        if len(valid) < self._sps:
            self._not_first = True
            return []

        # Phase search: run on EVERY block.
        #
        # WHY RE-SEARCH EVERY BLOCK:
        #   RTL-SDR (±20 ppm) and Pluto (±25 ppm) have independent clocks.
        #   Worst-case relative drift = 45 ppm × 2 MSPS = 90 samples/s.
        #   One callback = 262144/2e6 = 131 ms → 11.8 samples/callback drift.
        #   Eye opening = ±SPS/2 = ±4 samples.
        #   Without re-search: eye closes in just 0.3 callbacks ≈ 40 ms.
        #   With per-callback re-search: drift is bounded to ≤11.8 samples per
        #   callback, then immediately corrected.  The error per callback is at
        #   most 1 re-alignment glitch (bits in _pending in FrameSync), but
        #   since FrameSync is now a BIT-LEVEL sliding shift register it
        #   recovers automatically within 8 bits.  No "byte boundary" danger.
        #
        # Previous note ("Re-searching corrupts bit stream") was true with the
        # old byte-level FrameSync; that code is gone.
        best_phase = 0
        best_score = -1.0
        for ph in range(self._sps):
            sc = float(np.mean(np.abs(valid[ph::self._sps])))
            if sc > best_score:
                best_score = sc
                best_phase = ph
        self._phase = best_phase
        self._not_first = True

        # --- Sub-block sampling with periodic phase refinement ---------------
        # WHY SUB-BLOCK:
        #   Clock drift continues WITHIN a chunk.  At 45 ppm worst-case
        #   (RTL-SDR + Pluto), drift rate = 90 samples/s.  A full 262144-
        #   sample chunk (131 ms) accumulates 11.8 samples = 1.5 symbol
        #   periods of drift, which violates the ±0.5 symbol eye opening.
        #   By re-searching every SUB_BLOCK_SYMS symbols the maximum in-block
        #   drift is bounded:
        #     drift_per_sub = 90 × (SUB_BLOCK_SYMS×SPS/SR)
        #   SUB_BLOCK_SYMS=2048 → 8192 samples → 4.1 ms → 0.37 samples/sub
        #   → well within ±4 sample eye opening at all times.
        SUB_BLOCK_SYMS = 2048           # symbols between phase re-searches
        sub_samp       = SUB_BLOCK_SYMS * self._sps
        results = []
        pos = best_phase                # start from freshly searched phase
        while pos < len(valid):
            end = min(pos + sub_samp, len(valid))
            sub = valid[pos:end]
            # Re-search phase within this sub-block
            local_best = 0
            local_score = -1.0
            for ph in range(self._sps):
                sc = float(np.mean(np.abs(sub[ph::self._sps])))
                if sc > local_score:
                    local_score = sc
                    local_best = ph
            # Sample this sub-block
            for i in range(local_best, len(sub), self._sps):
                results.append((float(sub[i]), 0.0))
            # Advance: next sub-block starts right after this one
            pos = end
        return results



# ═══════════════════════════════════════════════════════════════
# 6. AGC  (Automatic Gain Control)
# ═══════════════════════════════════════════════════════════════
class AGC:
    """
    Feed-forward AGC normalised to the expected 4-FSK signal RMS.

    For 4-level FSK with uniform symbol probabilities:
        RMS = sqrt( ((-3)² + (-1)² + (+1)² + (+3)²) / 4 )
            = sqrt(5) ≈ 2.236

    Setting target=sqrt(5) preserves the natural {-3,-1,+1,+3} amplitude
    so the downstream slicer thresholds (−2, 0, +2) are always correct,
    independent of how many times push_iq() is called.

    IMPORTANT: target=1.0 (power normalisation) is WRONG here because it
    compresses +3/−3 symbols to ±1.34, placing them inside the ±2 range
    and causing the slicer to misidentify ~half of all symbols.

    init_power: seed = 5.0 (= target²) so there is zero transient on startup.
    """
    TARGET_RMS = float(np.sqrt(5.0))   # ≈ 2.236

    def __init__(self, target: float = TARGET_RMS,
                 alpha: float = 0.05,
                 init_power: float = 5.0):   # 5.0 = TARGET_RMS²
        self._target = target
        self._alpha  = alpha
        self._power  = init_power   # seed = true expected power → no transient

    def process(self, x: np.ndarray) -> np.ndarray:
        pwr = float(np.mean(x**2))
        if pwr > 1e-12:
            self._power = (1 - self._alpha) * self._power + self._alpha * pwr
        gain = self._target / (np.sqrt(self._power) + 1e-12)
        return x * gain


# ═══════════════════════════════════════════════════════════════
# 7. Symbol Slicer  (4-level decision)
# ═══════════════════════════════════════════════════════════════
_THRESHOLDS = (-2.0, 0.0, 2.0)   # midpoints between -3,-1,+1,+3

def slice_symbol(v: float) -> int:
    """Map analog level to nearest 4-FSK symbol {-3,-1,+1,+3}."""
    if   v < _THRESHOLDS[0]: return -3
    elif v < _THRESHOLDS[1]: return -1
    elif v < _THRESHOLDS[2]: return +1
    else:                     return +3

_SYM_TO_BITS = {-3: (0, 0), -1: (0, 1), +1: (1, 0), +3: (1, 1)}

def symbols_to_bits(symbols) -> list:
    bits = []
    for s in symbols:
        bits.extend(_SYM_TO_BITS[s])
    return bits

def bits_to_bytes_list(bits: list) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


# ═══════════════════════════════════════════════════════════════
# 8. CRC Functions
# ═══════════════════════════════════════════════════════════════
def crc8_rm(data: bytes) -> int:
    """CRC-8 (poly=0x31, init=0xFF) — RoboMaster header check."""
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc

def crc16_rm(data: bytes) -> int:
    """CRC-16/CCITT (poly=0x1021, init=0xFFFF) — RoboMaster frame check."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# ═══════════════════════════════════════════════════════════════
# 9. Protocol Parser
# ═══════════════════════════════════════════════════════════════
"""
Frame layout (little-endian integers):
  Offset  Size  Field
  ──────  ────  ──────────────────────────────
    0       1   SOF  = 0xA5
    1       2   DataLen   (bytes in payload+crc16 section)
    3       1   Seq
    4       1   CRC8  over bytes[0:4]
    5       2   CmdID  (0x0A01 … 0x0A06)
    7    varies Payload
   7+N      2   CRC16 over bytes[0 : 7+N]

Enemy Coordinates (0x0A01) — 24-byte payload:
  uint16 enemy_id
  float32 x, y, z     (metres)
  float32 vx, vy, vz  (m/s)
  uint16 flags
  pad 2 bytes → total 24

Interference Key (0x0A06) — 6-byte payload:
  char[6] key  (ASCII, null-padded)
"""

def _verify_frame(raw: bytes) -> bool:
    """Return True if CRC8 (header) and CRC16 (full frame minus last 2) pass."""
    if len(raw) < MIN_FRAME_LEN:
        return False
    if crc8_rm(raw[0:4]) != raw[4]:
        return False
    if crc16_rm(raw[:-2]) != struct.unpack_from('<H', raw, len(raw) - 2)[0]:
        return False
    return True


def parse_0x0A01(payload: bytes) -> dict:
    """
    Parse Enemy Coordinates (CmdID 0x0A01).
    Payload layout (24 bytes):
      uint16   enemy_id
      float32  x, y, z        (m)
      float32  vx, vy, vz     (m/s)
      uint16   flags
      uint8[2] _pad
    """
    if len(payload) < 24:
        raise ValueError(f"0x0A01 payload too short: {len(payload)} < 24")
    enemy_id,            = struct.unpack_from('<H',       payload, 0)
    x, y, z              = struct.unpack_from('<3f',      payload, 2)
    vx, vy, vz           = struct.unpack_from('<3f',      payload, 14)
    flags,               = struct.unpack_from('<H',       payload, 26)
    return {
        'cmd': '0x0A01',
        'enemy_id': enemy_id,
        'pos_m'   : (round(x,3), round(y,3), round(z,3)),
        'vel_ms'  : (round(vx,3), round(vy,3), round(vz,3)),
        'flags'   : f'0x{flags:04X}',
    }


def parse_0x0A06(payload: bytes) -> dict:
    """
    Parse Interference Key (CmdID 0x0A06).
    Payload layout (6 bytes):
      char[6]  key   (ASCII, zero-padded)
    """
    if len(payload) < 6:
        raise ValueError(f"0x0A06 payload too short: {len(payload)} < 6")
    key = payload[:6].rstrip(b'\x00').decode('ascii', errors='replace')
    return {
        'cmd': '0x0A06',
        'key': key,
    }


# Registry of all supported parsers
_PARSERS = {
    0x0A01: parse_0x0A01,
    0x0A06: parse_0x0A06,
}


def parse_frame(raw: bytes) -> dict | None:
    """
    Full frame parser.
    Returns a dict with fields, or None if CRC fails or unknown CmdID.
    """
    if not _verify_frame(raw):
        return None

    data_len,  = struct.unpack_from('<H', raw, 1)
    seq        = raw[3]
    cmd_id,    = struct.unpack_from('<H', raw, 5)  # CmdID is 2 bytes after CRC8

    # Payload starts at offset 7, ends before the last 2 CRC16 bytes
    payload = raw[7 : 7 + data_len - 2 - 2]       # DataLen includes CmdID(2)+payload+CRC16(2)
    # Corrected: payload window = raw[7 : len(raw)-2]
    payload = raw[7 : len(raw) - 2]

    parser = _PARSERS.get(cmd_id)
    if parser is None:
        return {'cmd': f'0x{cmd_id:04X}', 'raw_payload': payload.hex()}

    result = parser(payload)
    result['seq'] = seq
    return result


# ═══════════════════════════════════════════════════════════════
# 10. Bit-Stream Frame Synchroniser  (SOF search, no preamble)
# ═══════════════════════════════════════════════════════════════
class FrameSync:
    """
    SOF-hunting state machine operating at BIT level.

    WHY BIT-LEVEL:
      The previous byte-level implementation assumed SOF 0xA5 falls at a
      byte-aligned boundary in the bit stream.  In hardware, the phase
      search picks the best SAMPLING PHASE within one symbol period (0-7),
      but does NOT guarantee that the frame's SOF symbol is at a 4-symbol
      (= 1 byte) aligned position in the recovered stream.  Only 1 in 4
      alignments happens to be correct; the other 3 produce a SOF split
      across two consecutive bytes, so 0xA5 never appears and FrameSync
      never matches.

    ALGORITHM:
      Maintain a sliding 8-bit shift register.  After every new bit, check
      if the register equals 0xA5.  When it does, lock onto that bit
      alignment and accumulate the following bytes (header then body).
      All subsequent byte boundaries are aligned relative to this SOF bit.

    State machine:
        HUNT   → shift register scanning for 0xA5
        HEADER → accumulating bytes 1-4 (DataLen lo/hi, Seq, CRC8)
        BODY   → accumulating DataLen + 2 more bytes (CmdID+payload+CRC16)
    """
    def __init__(self, callback):
        """callback(dict) is called for every successfully decoded frame."""
        self._cb       = callback
        self._state    = 'HUNT'
        self._sreg     = 0           # 8-bit sliding shift register
        self._pending  = []          # bits for the current in-progress byte
        self._buf      = bytearray() # assembled bytes (starts with SOF 0xA5)
        self._need     = 0           # bytes remaining to complete BODY

    # ---- public interfaces -------------------------------------------------

    def feed_bits(self, bits):
        """Process an iterable of bits (0/1) one at a time."""
        for bit in bits:
            bit = int(bit)
            self._sreg = ((self._sreg << 1) | bit) & 0xFF

            if self._state == 'HUNT':
                if self._sreg == SOF:          # 0xA5 found at this bit alignment
                    self._buf     = bytearray([SOF])
                    self._pending = []         # next bit starts fresh byte
                    self._state   = 'HEADER'
            else:
                # Accumulate bits into bytes (8 bits → 1 byte, MSB first)
                self._pending.append(bit)
                if len(self._pending) == 8:
                    b = 0
                    for bv in self._pending:
                        b = (b << 1) | bv
                    self._pending = []
                    self._buf.append(b)

                    if self._state == 'HEADER':
                        if len(self._buf) == 5:   # SOF + DataLen(2) + Seq + CRC8
                            if crc8_rm(bytes(self._buf[:4])) != self._buf[4]:
                                self._state = 'HUNT'
                                self._buf.clear()
                            else:
                                data_len = struct.unpack_from('<H', self._buf, 1)[0]
                                if data_len < 4 or data_len > 512:
                                    self._state = 'HUNT'
                                    self._buf.clear()
                                else:
                                    self._need  = data_len + 2
                                    self._state = 'BODY'

                    elif self._state == 'BODY':
                        self._need -= 1
                        if self._need == 0:
                            result = parse_frame(bytes(self._buf))
                            if result:
                                self._cb(result)
                            self._state = 'HUNT'
                            self._buf.clear()

    def feed_bytes(self, data: bytes):
        """Compatibility wrapper — converts bytes to bits and calls feed_bits."""
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        self.feed_bits(bits)


# ═══════════════════════════════════════════════════════════════
# 11. Full RX Pipeline  (top-level, wires all blocks together)
# ═══════════════════════════════════════════════════════════════
class RX4FSKPipeline:
    """
    End-to-end 4-RRC-FSK receiver pipeline.

    IQ samples (from RTL-SDR callback or simulation)
        ↓  PreFilter  (300 kHz LPF, removes OOB noise: +5 dB SNR)
        ↓  FM-demod   (conjugate-delay + DC-AFC for carrier offset)
        ↓  RRC matched filter  (88-tap)
        ↓  AGC
        ↓  M&M TED  (symbol timing)
        ↓  Slicer  (4-level → {-3,-1,+1,+3})
        ↓  bits → bytes
        ↓  FrameSync (SOF hunt + CRC validate)
        ↓  parse_frame  → decoded fields

    Usage:
        pipe = RX4FSKPipeline(on_frame=print)
        pipe.push_iq(iq_array)   # call repeatedly with new IQ data
    """
    def __init__(self, on_frame=None):
        self._lpf  = PreFilter()
        self._fmd  = FMDemod()
        self._mf   = MatchedFilter(RRC_KERNEL)
        self._agc  = AGC()
        self._ted  = GardnerTED(SPS, initial_offset=TOTAL_CHAIN_DELAY)
        self._sync = FrameSync(on_frame or self._default_cb)
        # _bit_buf removed — bit-level FrameSync handles any byte alignment

    @staticmethod
    def _default_cb(frame: dict):
        print("[FRAME]", frame)

    def push_iq(self, iq: np.ndarray):
        """Feed complex IQ samples (dtype=complex64 or complex128)."""
        # 1. IQ bandpass pre-filter (LPF 300 kHz, reduces OOB noise)
        iq_filt = self._lpf.run(iq)

        # 2. FM demodulation → normalised frequency samples (+DC-AFC)
        freq_norm = self._fmd.demod(iq_filt)

        # 3. RRC matched filter
        mf_out = self._mf.run(freq_norm)

        # 4. AGC
        mf_agc = self._agc.process(mf_out)

        # 5. Gardner TED → symbol decisions
        ted_results = self._ted.process(mf_agc)

        # 6. Slice → bits, feed directly into bit-level FrameSync.
        # No byte-alignment assumption: FrameSync uses sliding 8-bit shift
        # register to detect SOF at ANY bit boundary (fixes hardware issue
        # where the start-of-frame symbol is not at a 4-symbol byte boundary).
        bits = []
        for sym_val, _err in ted_results:
            sym = slice_symbol(sym_val)
            bits.extend(_SYM_TO_BITS[sym])
        self._sync.feed_bits(bits)

    @property
    def freq_offset_hz(self) -> float:
        return self._fmd.freq_offset_hz


# ═══════════════════════════════════════════════════════════════
# 12.  Pluto TX  (ADALM-PLUTO transmitter, pyadi-iio)
# ═══════════════════════════════════════════════════════════════
class PlutoTX:
    """
    Transmit a 4-RRC-FSK protocol frame continuously using ADALM-PLUTO
    in cyclic-buffer mode.

    Usage:
        from fsk_digital_twin import build_protocol_frame, bytes_to_bits, bits_to_symbols
        tx = PlutoTX(key="RM2026")
        tx.start()        # non-blocking, runs in background thread
        ...               # RTL-SDR receiving
        tx.stop()
    """
    def __init__(self, key: str = "RM2026",
                 uri: str = PLUTO_URI,
                 attenuation: float = TX_ATTENUATION):
        self._key   = key
        self._uri   = uri
        self._atten = attenuation
        self._sdr   = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # -------- build IQ buffer ------------------------------------------------
    @staticmethod
    def _build_iq(key: str) -> np.ndarray:
        """Build the cyclic IQ waveform for one frame (repeated 4× for glitch-free loop).

        Returns complex64 array scaled to Pluto DAC range (±2^14).
        pyadi-iio sdr.tx() expects complex float (real+1j*imag) where the
        magnitude is in DAC counts.  Sending raw int16 interleaved is wrong.
        """
        try:
            from fsk_digital_twin import (build_protocol_frame,
                                           bytes_to_bits, bits_to_symbols)
        except ImportError as ex:
            raise ImportError("fsk_digital_twin.py must be in the same folder") from ex

        frame = build_protocol_frame(key)
        syms  = bits_to_symbols(bytes_to_bits(frame))
        # 4× repeat so the cyclic restart doesn't create a phase glitch
        syms_rep = np.tile(syms, 4)

        upsampled = np.zeros(len(syms_rep) * SPS)
        upsampled[::SPS] = syms_rep
        from scipy.signal import fftconvolve
        fp_full = fftconvolve(upsampled, RRC_KERNEL, mode='full')
        delay   = len(RRC_KERNEL) // 2
        fp      = fp_full[delay: delay + len(upsampled)]
        freq_hz = fp * (FSK_DEVIATION / 3.0)
        phase   = np.cumsum(freq_hz) * 2 * np.pi / SAMPLE_RATE
        # pyadi-iio sdr.tx() expects complex128/64 where values are in DAC counts
        # (same scale as the official pyadi-iio example: i = cos(...)*2**14)
        iq = np.exp(1j * phase).astype(np.complex64) * (2**14 * 0.5)
        return iq.astype(np.complex64)

    # -------- hardware init --------------------------------------------------
    def _open_pluto(self):
        try:
            import adi
        except ImportError:
            raise ImportError("pyadi-iio not installed.  Run: pip install pyadi-iio")

        print(f"[PLUTO-TX] Connecting to {self._uri} ...")
        sdr = adi.Pluto(self._uri)

        sdr.sample_rate          = int(SAMPLE_RATE)
        sdr.tx_rf_bandwidth      = int(SAMPLE_RATE)
        sdr.tx_lo                = int(CENTER_FREQ)
        sdr.tx_hardwaregain_chan0 = self._atten   # e.g. −50 dB
        sdr.tx_cyclic_buffer     = TX_CYCLIC

        print(f"[PLUTO-TX] {CENTER_FREQ/1e6:.3f} MHz  "
              f"SR={SAMPLE_RATE/1e6:.1f} MSPS  "
              f"Atten={self._atten} dB")
        return sdr

    # -------- public API -----------------------------------------------------
    def start(self):
        """Open Pluto, push IQ buffer, start cyclic TX in background thread."""
        self._stop_evt.clear()
        iq_buf = self._build_iq(self._key)
        self._sdr = self._open_pluto()
        self._sdr.tx(iq_buf)
        print(f"[PLUTO-TX] Cyclic TX running  "
              f"({len(iq_buf)} samples / frame repeat x4)")

        # Keep thread alive (Pluto cyclic TX is self-sustaining;
        # thread only monitors the stop event)
        def _keepalive():
            self._stop_evt.wait()

        self._thread = threading.Thread(target=_keepalive, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop cyclic TX and release Pluto."""
        self._stop_evt.set()
        if self._sdr:
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None
        print("[PLUTO-TX] Stopped.")


# ═══════════════════════════════════════════════════════════════
# 13.  RTL-SDR RX  (pyrtlsdr, local DLL auto-loaded at module top)
# ═══════════════════════════════════════════════════════════════
def run_rtlsdr(on_frame=None):
    """
    Open RTL-SDR V4, start async IQ capture, push into RX4FSKPipeline.
    Press Ctrl-C to stop.

    librtlsdr.dll + libusb-1.0.dll must reside next to this script.
    The os.add_dll_directory() call at module-level ensures they are found.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        print("[ERROR] pyrtlsdr not installed.")
        print("        Run:  pip install pyrtlsdr")
        print(f"        DLLs present: "
              f"librtlsdr.dll={'Y' if os.path.exists(os.path.join(_SCRIPT_DIR,'librtlsdr.dll')) else 'N'}, "
              f"libusb-1.0.dll={'Y' if os.path.exists(os.path.join(_SCRIPT_DIR,'libusb-1.0.dll')) else 'N'}")
        return

    try:
        sdr = RtlSdr()
    except Exception as exc:
        print(f"[ERROR] Cannot open RTL-SDR: {exc}")
        return

    sdr.sample_rate    = SAMPLE_RATE
    sdr.center_freq    = CENTER_FREQ
    sdr.gain           = GAIN
    # RTL-SDR V4: set_freq_correction(0) raises LIBUSB_ERROR_INVALID_PARAM
    # Only apply correction when actually needed (non-zero ppm)
    _PPM_CORRECTION = 0   # change if your dongle needs calibration
    if _PPM_CORRECTION != 0:
        sdr.freq_correction = _PPM_CORRECTION

    print(f"[RTL-SDR]  Tuned to {CENTER_FREQ/1e6:.3f} MHz  "
          f"SR={SAMPLE_RATE/1e6:.1f} MSPS  Gain={GAIN} dB")
    print("[RTL-SDR]  Receiving…  (Ctrl-C to stop)")

    _cb_count   = [0]
    _frame_count = [0]
    _orig_on_frame = on_frame

    def _on_frame_counted(d):
        _frame_count[0] += 1
        if _orig_on_frame:
            _orig_on_frame(d)

    pipe = RX4FSKPipeline(_on_frame_counted)     # re-create with counting wrapper

    def _callback(samples, _sdr):
        # pyrtlsdr returns complex128 normalised to [-1, +1]
        iq = samples.astype(np.complex64)
        pipe.push_iq(iq)
        _cb_count[0] += 1
        if _cb_count[0] % 8 == 0:   # ~every 1 second
            iq_rms  = float(np.sqrt(np.mean(np.abs(iq)**2)))
            fll_hz  = pipe.freq_offset_hz
            ted_ph  = pipe._ted._phase
            print(f"[DBG cb={_cb_count[0]:3d}] "
                  f"IQ_RMS={iq_rms:.4f}  AFC={fll_hz:+.0f} Hz  "
                  f"phase={ted_ph}  frames={_frame_count[0]}", flush=True)

    try:
        sdr.read_samples_async(_callback, num_samples=BUF_SAMPLES)
    except KeyboardInterrupt:
        pass
    finally:
        sdr.close()
        print("\n[RTL-SDR]  Stopped.")


# ═══════════════════════════════════════════════════════════════
# 13. Software Loopback Test  (validates pipeline without hardware)
# ═══════════════════════════════════════════════════════════════
def _tx_4fsk(symbols: np.ndarray) -> np.ndarray:
    """Minimal TX chain (mirroring fsk_digital_twin.py) for self-test."""
    upsampled = np.zeros(len(symbols) * SPS)
    upsampled[::SPS] = symbols
    freq_pulse = np.convolve(upsampled, RRC_KERNEL, mode='full')
    delay = len(RRC_KERNEL) // 2
    freq_pulse = freq_pulse[delay: delay + len(upsampled)]
    freq_hz    = freq_pulse * (FSK_DEVIATION / 3.0)   # ← CORRECT formula
    phase      = np.cumsum(freq_hz) * 2 * np.pi / SAMPLE_RATE
    return np.exp(1j * phase).astype(np.complex64) * 0.5


def software_loopback_test():
    """
    Realistic hardware-equivalent loopback test.

    Conditions designed to match true competition/hardware scenario:
      • freq_offset=+5000 Hz  (matches observed RTL-SDR crystal error)
      • SNR=15 dB             (conservative over-the-air estimate)
      • Streaming simulation: IQ fed in BUF_SAMPLES-sized chunks,
        exactly as RTL-SDR async callback delivers data.  This tests
        AFC convergence across blocks and bit-level FrameSync across
        chunk boundaries — the exact failure modes seen in hardware.
      • Frame repeated 20× (≈2.4s of continuous TX, like Pluto cyclic mode)

    Pass condition: ≥1 frame decoded with correct key.
    """
    from fsk_digital_twin import (build_protocol_frame, bytes_to_bits,
                                   bits_to_symbols)

    print("=" * 58)
    print("Software Loopback Test  —  Realistic Hardware Conditions")
    print("  freq_offset=+5000 Hz,  SNR=15 dB,  streaming chunks")
    print("=" * 58)

    key   = "RM2026"
    frame = build_protocol_frame(key)
    bits  = bytes_to_bits(frame)
    syms  = bits_to_symbols(bits)
    print(f"TX Frame ({len(frame)}B): {frame.hex(' ').upper()}")
    print(f"  Symbol mean={np.mean(syms):.3f}  (non-zero → old FM-AFC would fail)")

    # Build continuous TX stream: 5000 frame repetitions ≈ 25 s
    # 5000 × 60 sym × 8 sps = 2,400,000 samples ≈ 9 BUF_SAMPLES chunks
    # This ensures multiple cross-chunk boundaries are exercised.
    REPS = 5000
    syms_rep = np.tile(syms, REPS)
    tx_iq    = _tx_4fsk(syms_rep)

    # Channel: freq offset + AWGN at 15 dB SNR
    freq_offset_hz = 5000
    snr_db         = 15.0
    sig_power  = float(np.mean(np.abs(tx_iq)**2))
    noise_pwr  = sig_power / (10 ** (snr_db / 10))
    rng        = np.random.default_rng(42)
    noise      = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(len(tx_iq)) + 1j * rng.standard_normal(len(tx_iq))
    )
    t_arr  = np.arange(len(tx_iq), dtype=np.float64) / SAMPLE_RATE
    rx_iq  = (tx_iq * np.exp(1j * 2 * np.pi * freq_offset_hz * t_arr)
              + noise).astype(np.complex64)

    received = []

    def record(d):
        received.append(d)
        print(f"  → Decoded: {d}")

    pipe  = RX4FSKPipeline(on_frame=record)
    # Use hardware-identical chunk size to test cross-chunk bit reassembly
    CHUNK = BUF_SAMPLES
    n_chunks = 0
    for start in range(0, len(rx_iq), CHUNK):
        pipe.push_iq(rx_iq[start : start + CHUNK])
        n_chunks += 1

    print(f"\n  Chunks processed: {n_chunks}  ({n_chunks*CHUNK/SAMPLE_RATE*1000:.0f} ms total)")
    print(f"  AFC converged to: {pipe.freq_offset_hz:+.0f} Hz  (true offset={freq_offset_hz:+d} Hz)")
    ok = any(r.get('key') == key for r in received)
    if ok:
        print(f"[PASS] Key correctly recovered ({len(received)} frame(s) decoded).")
    else:
        print(f"[FAIL] Key not recovered — decoded {len(received)} frame(s).")
    return received


# ─────────────────────────── Entry Point ───────────────────────
if __name__ == '__main__':
    """
    Usage:
      python rx_4fsk_pipeline.py                # software loopback self-test
      python rx_4fsk_pipeline.py --hw           # Pluto TX + RTL-SDR RX
      python rx_4fsk_pipeline.py --hw --rx-only # RTL-SDR RX only (external TX)
      python rx_4fsk_pipeline.py --hw --tx-only # Pluto TX only   (dry run)
      python rx_4fsk_pipeline.py --hw --key RM2026  # custom key
      python rx_4fsk_pipeline.py --capture      # Save 2s IQ to iq_capture.npy
      python rx_4fsk_pipeline.py --diagnose     # Analyse iq_capture.npy offline
    """
    hw       = '--hw'       in sys.argv
    rx_only  = '--rx-only'  in sys.argv
    tx_only  = '--tx-only'  in sys.argv
    capture  = '--capture'  in sys.argv
    diagnose = '--diagnose' in sys.argv

    key = 'RM2026'
    if '--key' in sys.argv:
        idx = sys.argv.index('--key')
        if idx + 1 < len(sys.argv):
            key = sys.argv[idx + 1]

    # ── Offline IQ diagnostic ────────────────────────────────────
    if diagnose:
        cap_file = 'iq_capture.npy'
        if not os.path.exists(cap_file):
            print(f"[ERROR] {cap_file} not found. Run with --capture first.")
            sys.exit(1)

        iq = np.load(cap_file)
        print(f"[DIAG] Loaded {len(iq)} samples ({len(iq)/SAMPLE_RATE*1000:.0f} ms)")
        print(f"       IQ amplitude: mean={np.mean(np.abs(iq)):.4f}  "
              f"max={np.max(np.abs(iq)):.4f}  RMS={np.sqrt(np.mean(np.abs(iq)**2)):.4f}")

        # Apply pre-filter then FM demod
        lpf = PreFilter()
        iq_filt = lpf.run(iq.astype(np.complex64))
        fmd = FMDemod()
        freq_norm = fmd.demod(iq_filt)
        print(f"       FM demod (after 300kHz LPF + AFC): "
              f"mean={np.mean(freq_norm):.3f}  "
              f"std={np.std(freq_norm):.3f}  "
              f"max|val|={np.max(np.abs(freq_norm)):.3f}")
        print(f"       Fraction outside ±4: "
              f"{100*np.mean(np.abs(freq_norm)>4):.1f}%  (expect <5% for good signal)")
        print(f"       AFC DC estimate: {fmd.freq_offset_hz:+.0f} Hz")

        # MF + AGC
        mf  = MatchedFilter(RRC_KERNEL)
        agc = AGC()
        mf_agc = agc.process(mf.run(freq_norm))
        print(f"       After MF+AGC: RMS={np.sqrt(np.mean(mf_agc**2)):.3f}  "
              f"(target≈{AGC.TARGET_RMS:.3f})")

        # M&M TED symbol sampling
        ted = GardnerTED(SPS, initial_offset=TOTAL_CHAIN_DELAY)
        results = ted.process(mf_agc)
        print(f"       M&M TED: {len(results)} symbols recovered")

        sym_hist = {-3: 0, -1: 0, 1: 0, 3: 0}
        for v, _ in results:
            sym_hist[slice_symbol(v)] += 1
        total = sum(sym_hist.values())
        print(f"       Symbol distribution (expect ~25% each for random data):")
        for k, v in sym_hist.items():
            print(f"         {k:+d}: {100*v/max(total,1):.1f}%")

        # Frame sync attempt
        bit_buf = []
        for v, _ in results:
            bit_buf.extend(_SYM_TO_BITS[slice_symbol(v)])
        raw_bytes = bits_to_bytes_list(bit_buf)
        received = []
        FrameSync(received.append).feed_bytes(raw_bytes)
        print(f"\n       Frames decoded: {len(received)}")
        for r in received[:5]:
            print(f"         {r}")

        # Spectrum with DC-notch: check carrier offset
        n_fft = min(65536, len(iq))
        spec = np.abs(np.fft.fftshift(np.fft.fft(iq[:n_fft])))**2
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1/SAMPLE_RATE))
        # Null the DC bin (RTL-SDR DC spike)
        dc_bins = int(n_fft * 5000 / SAMPLE_RATE)   # ±5 kHz around DC
        spec[n_fft//2 - dc_bins : n_fft//2 + dc_bins] = 0
        peak_idx = np.argmax(spec)
        carrier_offset_hz = freqs[peak_idx]
        print(f"\n       Strongest spectral component (DC-notched): "
              f"{carrier_offset_hz/1e3:.1f} kHz offset from center")

        # Re-demod after compensating for carrier offset
        if abs(carrier_offset_hz) > 1e3:
            t = np.arange(len(iq)) / SAMPLE_RATE
            iq_shifted = iq * np.exp(-1j * 2 * np.pi * carrier_offset_hz * t)
            lpf2 = PreFilter()
            fmd2 = FMDemod()
            fn2 = fmd2.demod(lpf2.run(iq_shifted.astype(np.complex64)))
            print(f"       FM demod AFTER {carrier_offset_hz/1e3:.1f} kHz shift: "
                  f"mean={np.mean(fn2):.3f}  std={np.std(fn2):.3f}  "
                  f"outside\u00b14: {100*np.mean(np.abs(fn2)>4):.1f}%")
            mf2  = MatchedFilter(RRC_KERNEL)
            agc2 = AGC()
            mfa2 = agc2.process(mf2.run(fn2))
            ted2 = GardnerTED(SPS, initial_offset=TOTAL_CHAIN_DELAY)
            res2 = ted2.process(mfa2)
            bb2  = []
            for v2, _ in res2:
                bb2.extend(_SYM_TO_BITS[slice_symbol(v2)])
            rb2 = bits_to_bytes_list(bb2)
            rec2 = []
            FrameSync(rec2.append).feed_bytes(rb2)
            print(f"       Frames decoded after shift: {len(rec2)}")
            for r in rec2[:5]:
                print(f"         {r}")
        else:
            print("       Carrier offset negligible (<1 kHz), no shift applied.")

        sys.exit(0)

    # ── Capture 2 seconds of RTL-SDR IQ ─────────────────────────
    if capture:
        pluto = None
        if hw:
            pluto = PlutoTX(key=key)
            try:
                pluto.start()
            except Exception as exc:
                print(f"[ERROR] Pluto TX failed: {exc}"); sys.exit(1)

        try:
            from rtlsdr import RtlSdr
        except ImportError:
            print("[ERROR] pyrtlsdr not installed."); sys.exit(1)

        sdr = RtlSdr()
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = CENTER_FREQ
        sdr.gain        = GAIN
        _PPM_CORRECTION = 0
        if _PPM_CORRECTION != 0:
            sdr.freq_correction = _PPM_CORRECTION

        n_capture = int(2 * SAMPLE_RATE)   # 2 seconds
        print(f"[CAPTURE] Capturing {n_capture} samples ({n_capture/SAMPLE_RATE:.1f}s)...")
        iq_raw = sdr.read_samples(n_capture)
        sdr.close()
        np.save('iq_capture.npy', iq_raw.astype(np.complex64))
        print(f"[CAPTURE] Saved to iq_capture.npy  "
              f"(IQ RMS={np.sqrt(np.mean(np.abs(iq_raw)**2)):.4f})")
        if pluto:
            pluto.stop()
        print("[CAPTURE] Run with --diagnose to analyse.")
        sys.exit(0)

    if not hw:
        software_loopback_test()
        sys.exit(0)

    # ── Hardware streaming mode ───────────────────────────────────
    def _on_frame(d: dict):
        import json, datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[{ts}] {json.dumps(d, ensure_ascii=False)}", flush=True)

    pluto = None
    if not rx_only:
        pluto = PlutoTX(key=key)
        try:
            pluto.start()
        except Exception as exc:
            print(f"[ERROR] Pluto TX failed: {exc}")
            sys.exit(1)

    if not tx_only:
        try:
            run_rtlsdr(on_frame=_on_frame)
        finally:
            if pluto:
                pluto.stop()
    else:
        print("[TX-ONLY] Pluto broadcasting. Press Ctrl-C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if pluto:
                pluto.stop()
