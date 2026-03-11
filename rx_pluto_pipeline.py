#!/usr/bin/env python3
"""
4-RRC-FSK Dual-Pluto Pipeline  —  RoboMaster 2026 Radar EW
===========================================================
TX Hardware : ADALM-PLUTO (URI from config.json → pluto_uri)     (pyadi-iio)
RX Hardware : ADALM-PLUTO (URI from config.json → pluto_rx_uri)  (pyadi-iio)  + filter+amp
Modulation  : 4-RRC-FSK, NO PREAMBLE

This file is derived from rx_4fsk_pipeline.py (which ran successfully with
RTL-SDR as RX).  The ENTIRE DSP chain is kept identically; only the
hardware driver layer is changed from RTL-SDR to a second ADALM-PLUTO.

Physical-Layer Math
-------------------
  Δf  = Symbol Rate = 250 kHz
  SPS = SR / Baud  = 2.5e6 / 250e3  = 10
  BW  = 2*(Δf + 0.02 MHz) = 0.54 MHz
  f(s) = s * (Δf/3),  s ∈ {-3,-1,+1,+3}

DSP Chain  (identical to rx_4fsk_pipeline.py)
----------------------------------------------
  Pluto-RX → PreFilter(LPF 300 kHz) → FMDemod → RRC-MF(88 tap) → AGC →
  GardnerTED → Slicer → FrameSync(bit-level) → parse_frame → callback

Dependencies
------------
  pip install numpy scipy pyadi-iio

Usage
-----
  python rx_pluto_pipeline.py              # software loopback self-test
  python rx_pluto_pipeline.py --hw         # Pluto TX + Pluto RX
  python rx_pluto_pipeline.py --hw --rx-only
  python rx_pluto_pipeline.py --hw --tx-only
  python rx_pluto_pipeline.py --hw --key RM2026
"""

import os
import sys
import struct
import threading
import queue
import time

import numpy as np
from scipy.signal import lfilter, firwin

# ─────────────────────────── Config ────────────────────────────
SAMPLE_RATE     = 2_500_000    # 2.5 MSPS  (Pluto ADC, matches config.json L0)
BAUD_RATE       = 250_000      # 250 kBaud
SPS             = SAMPLE_RATE // BAUD_RATE   # = 10

# RM2026 对手广播频率表（单位 Hz）
_BROADCAST_FREQ: dict[str, float] = {
    "blue": 433_920_000.0,   # 蓝队广播
    "red":  433_200_000.0,   # 红队广播
}

# Pluto URIs + 频率 + 增益 — 全部从 config.json 读取
def _load_config() -> tuple[str, str, int, int, int]:
    """
    从 config.json 读取，返回 (tx_uri, rx_uri, tx_freq_hz, rx_freq_hz, rx_gain_db)

    TX_FREQ = 我方队伍的广播频率（我们发射到这个频率）
    RX_FREQ = 对手队伍的广播频率（我们收听这个频率）

      team_color=red  → TX=433.200 MHz, RX=433.920 MHz
      team_color=blue → TX=433.920 MHz, RX=433.200 MHz
    """
    import json, pathlib
    _cfg = pathlib.Path(__file__).parent / "config.json"
    try:
        raw = json.loads(_cfg.read_text(encoding="utf-8"))
        tx_uri = raw.get("pluto_uri") or "ip:192.168.2.1"
        rx_uri = raw.get("pluto_rx_uri") or tx_uri
        our_color = str(raw.get("team_color", "red")).lower().strip()
        opponent  = "red" if our_color == "blue" else "blue"
        tx_freq = int(_BROADCAST_FREQ.get(our_color,  433_920_000.0))  # 我方信道（TX 用）
        rx_freq = int(_BROADCAST_FREQ.get(opponent,   433_200_000.0))  # 对方信道（RX 用）
        gain    = int(raw.get("rx_gain_db", 50))
        print(f"[CONFIG] team={our_color.upper()}  "
              f"TX={tx_freq/1e6:.3f} MHz  RX(listen)={rx_freq/1e6:.3f} MHz  "
              f"gain={gain} dB  RX_URI={rx_uri}")
        return str(tx_uri), str(rx_uri), tx_freq, rx_freq, gain
    except Exception as e:
        print(f"[CONFIG] 读取 config.json 失败 ({e})，使用默认值")
        return "ip:192.168.2.1", "ip:192.168.2.1", 433_920_000, 433_200_000, 50

PLUTO_TX_URI, PLUTO_RX_URI, TX_CENTER_FREQ, CENTER_FREQ, RX_GAIN_DB = _load_config()

TX_ATTENUATION  = 0            # dB attenuation on TX Pluto (0 = max power)
TX_CYCLIC       = True

FSK_DEVIATION   = 250_000      # Δf = 250 kHz
RRC_ALPHA       = 0.25
RRC_SPAN        = 11           # → 11*SPS taps
AFC_ALPHA       = 0.0          # AFC disabled (same reason as RTL-SDR version)

# Protocol constants
SOF        = 0xA5
CMDS_LEN   = {
    0x0A01: 24, 0x0A02: 12, 0x0A03: 10,
    0x0A04:  8, 0x0A05: 36, 0x0A06:  6,
}
MIN_FRAME_LEN = 9   # SOF(1)+DataLen(2)+Seq(1)+CRC8(1)+CmdID(2)+CRC16(2)

BUF_SAMPLES = 256 * 1024   # samples per capture block (~102 ms at 2.5 MSPS)


# ═══════════════════════════════════════════════════════════════
# 1. RRC Filter
# ═══════════════════════════════════════════════════════════════
def rrc_filter(alpha: float, span: int, sps: int) -> np.ndarray:
    n_taps = span * sps
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
# 2. FM Demodulator  (stateful – preserves last sample across blocks)
# ═══════════════════════════════════════════════════════════════
class FMDemod:
    """
    Conjugate-delay FM discriminator.
    Normalised so ideal symbol levels → {-3,-1,+1,+3}.
    self._prev preserves the last IQ sample across consecutive blocks.
    AFC disabled (same rationale as rx_4fsk_pipeline.py).
    """
    def __init__(self):
        self._prev = complex(1, 0)

    def demod(self, iq: np.ndarray) -> np.ndarray:
        iq_prev = np.empty_like(iq)
        iq_prev[0] = self._prev
        iq_prev[1:] = iq[:-1]
        self._prev = iq[-1]
        d_phase = np.angle(iq * np.conj(iq_prev))
        f_hz = d_phase * (SAMPLE_RATE / (2 * np.pi))
        return (f_hz / (FSK_DEVIATION / 3.0)).astype(np.float32)

    @property
    def freq_offset_hz(self) -> float:
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 2b. IQ Pre-filter  (300 kHz LPF, stateful)
# ═══════════════════════════════════════════════════════════════
_LPF_CUTOFF_HZ = 300_000
_LPF_TAPS      = 63
_LPF_KERNEL    = firwin(_LPF_TAPS, cutoff=_LPF_CUTOFF_HZ,
                        fs=SAMPLE_RATE).astype(np.float32)

class PreFilter:
    """63-tap LPF applied to complex IQ before FM demod.  Stateful (lfilter zi)."""
    def __init__(self):
        self._zi = np.zeros(len(_LPF_KERNEL) - 1, dtype=np.complex64)

    def run(self, iq: np.ndarray) -> np.ndarray:
        out, self._zi = lfilter(_LPF_KERNEL, 1.0,
                                iq.astype(np.complex64), zi=self._zi)
        return out.astype(np.complex64)


# ═══════════════════════════════════════════════════════════════
# 3. Matched Filter  (stateful overlap-save)
# ═══════════════════════════════════════════════════════════════
class MatchedFilter:
    """Stateful RRC matched filter (overlap-save, correct cross-block continuity)."""
    def __init__(self, kernel: np.ndarray):
        self._k     = kernel
        self._state = np.zeros(len(kernel) - 1, dtype=np.float32)

    def run(self, x: np.ndarray) -> np.ndarray:
        padded = np.concatenate([self._state, x])
        out    = np.convolve(padded, self._k, mode='valid')
        self._state = padded[-(len(self._k)-1):]
        return out


MF_GROUP_DELAY   = (RRC_SPAN * SPS - 1) // 2
LPF_GROUP_DELAY  = (_LPF_TAPS - 1) // 2
TOTAL_CHAIN_DELAY = LPF_GROUP_DELAY + MF_GROUP_DELAY


# ═══════════════════════════════════════════════════════════════
# 4. AGC
# ═══════════════════════════════════════════════════════════════
class AGC:
    """Feed-forward AGC targeting sqrt(5) ≈ 2.236 RMS (natural 4-FSK level)."""
    TARGET_RMS = float(np.sqrt(5.0))

    def __init__(self, target: float = TARGET_RMS,
                 alpha: float = 0.05,
                 init_power: float = 5.0):
        self._target = target
        self._alpha  = alpha
        self._power  = init_power   # seed = expected power → no startup transient

    def process(self, x: np.ndarray) -> np.ndarray:
        pwr = float(np.mean(x**2))
        if pwr > 1e-12:
            self._power = (1 - self._alpha) * self._power + self._alpha * pwr
        gain = self._target / (np.sqrt(self._power) + 1e-12)
        return x * gain


# ═══════════════════════════════════════════════════════════════
# 5. Gardner TED  (blind phase search, sub-block re-search)
# ═══════════════════════════════════════════════════════════════
class GardnerTED:
    """
    Adaptive blind-phase symbol sampler from rx_4fsk_pipeline.py.
    Phase re-searched every SUB_BLOCK_SYMS symbols to track Pluto-to-Pluto
    clock drift (≤45 ppm combined).
    """
    def __init__(self, sps: int = SPS, kp=None, ki=None,
                 initial_offset: int = 0):
        self._sps       = sps
        self._skip_left = initial_offset
        self._phase     = 0
        self._not_first = False
        self._omega     = float(sps)
        self._next_offset = 0

    def process(self, samples: np.ndarray) -> list:
        arr = np.asarray(samples, dtype=np.float32)
        n   = len(arr)

        skip = min(self._skip_left, n)
        self._skip_left = max(0, self._skip_left - n)
        valid = arr[skip:]

        if len(valid) < self._sps:
            self._not_first = True
            return []

        # Initial phase search on full valid portion
        best_phase, best_score = 0, -1.0
        for ph in range(self._sps):
            sc = float(np.mean(np.abs(valid[ph::self._sps])))
            if sc > best_score:
                best_score = sc
                best_phase = ph
        self._phase     = best_phase
        self._not_first = True

        SUB_BLOCK_SYMS = 2048
        sub_samp       = SUB_BLOCK_SYMS * self._sps
        results = []
        pos = best_phase
        while pos < len(valid):
            end = min(pos + sub_samp, len(valid))
            sub = valid[pos:end]
            local_best, local_score = 0, -1.0
            for ph in range(self._sps):
                sc = float(np.mean(np.abs(sub[ph::self._sps])))
                if sc > local_score:
                    local_score = sc
                    local_best  = ph
            for i in range(local_best, len(sub), self._sps):
                results.append((float(sub[i]), 0.0))
            pos = end
        return results


# ═══════════════════════════════════════════════════════════════
# 6. Symbol Slicer
# ═══════════════════════════════════════════════════════════════
_THRESHOLDS  = (-2.0, 0.0, 2.0)
_SYM_TO_BITS = {-3: (0, 0), -1: (0, 1), +1: (1, 0), +3: (1, 1)}

def slice_symbol(v: float) -> int:
    if   v < _THRESHOLDS[0]: return -3
    elif v < _THRESHOLDS[1]: return -1
    elif v < _THRESHOLDS[2]: return +1
    else:                     return +3

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
# 7. CRC
# ═══════════════════════════════════════════════════════════════
def crc8_rm(data: bytes) -> int:
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc

def crc16_rm(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# ═══════════════════════════════════════════════════════════════
# 8. Protocol Parser
# ═══════════════════════════════════════════════════════════════
def _verify_frame(raw: bytes) -> bool:
    if len(raw) < MIN_FRAME_LEN:
        return False
    if crc8_rm(raw[0:4]) != raw[4]:
        return False
    if crc16_rm(raw[:-2]) != struct.unpack_from('<H', raw, len(raw) - 2)[0]:
        return False
    return True

def parse_0x0A06(payload: bytes) -> dict:
    if len(payload) < 6:
        raise ValueError(f"0x0A06 payload too short: {len(payload)}")
    key = payload[:6].rstrip(b'\x00').decode('ascii', errors='replace')
    return {'cmd': '0x0A06', 'key': key}

def parse_0x0A01(payload: bytes) -> dict:
    if len(payload) < 24:
        raise ValueError(f"0x0A01 payload too short: {len(payload)}")
    enemy_id,     = struct.unpack_from('<H',  payload, 0)
    x, y, z       = struct.unpack_from('<3f', payload, 2)
    vx, vy, vz    = struct.unpack_from('<3f', payload, 14)
    flags,        = struct.unpack_from('<H',  payload, 26)
    return {'cmd': '0x0A01', 'enemy_id': enemy_id,
            'pos_m': (round(x,3), round(y,3), round(z,3)),
            'vel_ms': (round(vx,3), round(vy,3), round(vz,3)),
            'flags': f'0x{flags:04X}'}

_PARSERS = {0x0A01: parse_0x0A01, 0x0A06: parse_0x0A06}

def parse_frame(raw: bytes) -> dict | None:
    if not _verify_frame(raw):
        return None
    data_len, = struct.unpack_from('<H', raw, 1)
    seq        = raw[3]
    cmd_id,    = struct.unpack_from('<H', raw, 5)
    payload    = raw[7 : len(raw) - 2]
    parser     = _PARSERS.get(cmd_id)
    if parser is None:
        return {'cmd': f'0x{cmd_id:04X}', 'raw_payload': payload.hex()}
    result = parser(payload)
    result['seq'] = seq
    return result


# ═══════════════════════════════════════════════════════════════
# 9. Bit-Stream Frame Synchroniser  (bit-level sliding SOF hunt)
# ═══════════════════════════════════════════════════════════════
class FrameSync:
    """
    Sliding 8-bit shift register hunts for SOF=0xA5 at any bit alignment.
    State preserved across feed_bits() calls.
    """
    def __init__(self, callback):
        self._cb      = callback
        self._state   = 'HUNT'
        self._sreg    = 0
        self._pending = []
        self._buf     = bytearray()
        self._need    = 0

    def feed_bits(self, bits):
        for bit in bits:
            bit = int(bit)
            self._sreg = ((self._sreg << 1) | bit) & 0xFF

            if self._state == 'HUNT':
                if self._sreg == SOF:
                    self._buf     = bytearray([SOF])
                    self._pending = []
                    self._state   = 'HEADER'
            else:
                self._pending.append(bit)
                if len(self._pending) == 8:
                    b = 0
                    for bv in self._pending:
                        b = (b << 1) | bv
                    self._pending = []
                    self._buf.append(b)

                    if self._state == 'HEADER':
                        if len(self._buf) == 5:
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


# ═══════════════════════════════════════════════════════════════
# 10. Full RX Pipeline
# ═══════════════════════════════════════════════════════════════
class RX4FSKPipeline:
    """
    End-to-end 4-RRC-FSK receiver.
    Call push_iq() repeatedly with successive IQ blocks.
    All DSP stages are stateful and correctly handle block boundaries.
    """
    def __init__(self, on_frame=None):
        self._lpf  = PreFilter()
        self._fmd  = FMDemod()
        self._mf   = MatchedFilter(RRC_KERNEL)
        self._agc  = AGC()
        self._ted  = GardnerTED(SPS, initial_offset=TOTAL_CHAIN_DELAY)
        self._sync = FrameSync(on_frame or self._default_cb)

    @staticmethod
    def _default_cb(frame: dict):
        print("[FRAME]", frame)

    def push_iq(self, iq: np.ndarray):
        """Feed complex IQ (dtype=complex64, normalised to ~±1)."""
        iq_filt   = self._lpf.run(iq)
        freq_norm = self._fmd.demod(iq_filt)
        mf_out    = self._mf.run(freq_norm)
        mf_agc    = self._agc.process(mf_out)
        ted_res   = self._ted.process(mf_agc)
        bits = []
        for sym_val, _err in ted_res:
            sym = slice_symbol(sym_val)
            bits.extend(_SYM_TO_BITS[sym])
        self._sync.feed_bits(bits)

    @property
    def freq_offset_hz(self) -> float:
        return self._fmd.freq_offset_hz


# ═══════════════════════════════════════════════════════════════
# 11. Pluto TX
# ═══════════════════════════════════════════════════════════════
class PlutoTX:
    """Transmit 4-RRC-FSK frame continuously via ADALM-PLUTO cyclic buffer."""

    def __init__(self, key: str = "RM2026",
                 uri: str = PLUTO_TX_URI,
                 attenuation: float = TX_ATTENUATION):
        self._key   = key
        self._uri   = uri
        self._atten = attenuation
        self._sdr   = None
        self._stop_evt = threading.Event()

    @staticmethod
    def _build_iq(key: str) -> np.ndarray:
        try:
            from fsk_digital_twin import (build_protocol_frame,
                                           bytes_to_bits, bits_to_symbols)
        except ImportError as ex:
            raise ImportError("fsk_digital_twin.py must be in the same folder") from ex

        frame    = build_protocol_frame(key)
        syms     = bits_to_symbols(bytes_to_bits(frame))
        syms_rep = np.tile(syms, 4)   # 4x repeat for glitch-free cyclic loop

        upsampled = np.zeros(len(syms_rep) * SPS)
        upsampled[::SPS] = syms_rep
        from scipy.signal import fftconvolve
        fp_full = fftconvolve(upsampled, RRC_KERNEL, mode='full')
        delay   = len(RRC_KERNEL) // 2
        fp      = fp_full[delay: delay + len(upsampled)]
        freq_hz = fp * (FSK_DEVIATION / 3.0)
        phase   = np.cumsum(freq_hz) * 2 * np.pi / SAMPLE_RATE
        # Pluto DMA buffer is 16-bit; 2^14 * 0.5 = 8192 ≈ −6 dBFS.
        # Using 2047 (12-bit full-scale) only utilises 12.5% of the 16-bit range
        # (−18 dBFS), making the transmitted signal 4× too weak.
        # Value matches the proven rx_4fsk_pipeline.py reference.
        iq      = np.exp(1j * phase).astype(np.complex64) * float(2**14 * 0.5)
        return iq.astype(np.complex64)

    def _open_pluto(self):
        import adi
        print(f"[PLUTO-TX] Connecting to {self._uri} ...")
        sdr = adi.Pluto(self._uri)
        sdr.sample_rate          = int(SAMPLE_RATE)
        sdr.tx_rf_bandwidth      = int(SAMPLE_RATE)
        sdr.tx_lo                = int(TX_CENTER_FREQ)
        sdr.tx_hardwaregain_chan0 = -abs(self._atten)
        sdr.tx_cyclic_buffer     = TX_CYCLIC
        print(f"[PLUTO-TX] {TX_CENTER_FREQ/1e6:.3f} MHz  SR={SAMPLE_RATE/1e6:.2f} MSPS")
        return sdr

    def start(self):
        self._stop_evt.clear()
        iq_buf    = self._build_iq(self._key)
        self._sdr = self._open_pluto()
        self._sdr.tx(iq_buf)
        print(f"[PLUTO-TX] Cyclic TX running  ({len(iq_buf)} samples)")

        def _keepalive():
            self._stop_evt.wait()
        threading.Thread(target=_keepalive, daemon=True).start()

    def stop(self):
        self._stop_evt.set()
        if self._sdr:
            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            self._sdr = None
        print("[PLUTO-TX] Stopped.")


# ═══════════════════════════════════════════════════════════════
# 12. Pluto RX  (replaces RTL-SDR RX from rx_4fsk_pipeline.py)
# ═══════════════════════════════════════════════════════════════
def run_pluto_rx(on_frame=None, rx_gain_db: int = RX_GAIN_DB,
                 rx_uri: str = PLUTO_RX_URI):
    """
    Open the RX ADALM-PLUTO, pull IQ blocks in a polling loop, and feed
    them into RX4FSKPipeline.

    IQ normalization:
      Pluto ADC returns raw counts (12-bit, ±2047).  Dividing by 2048
      gives the same ±1 scale that pyrtlsdr delivers, so the DSP chain
      needs no modification.
    """
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed.  Run: pip install pyadi-iio")
        return

    print(f"[PLUTO-RX] Connecting to {rx_uri} ...")
    sdr = adi.Pluto(rx_uri)

    # ── 终止残留循环 TX ────────────────────────────────────────────────────
    # Pluto 固件的循环 DMA 在 Python 进程异常退出后仍会持续发射。
    # 每次启动 RX 前先强制销毁 TX buffer 并关闭增益，防止本机信号污染接收。
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_hardwaregain_chan0 = -89   # TX 完全静音

    sdr.rx_lo                   = int(CENTER_FREQ)
    sdr.sample_rate             = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth         = int(SAMPLE_RATE)
    sdr.rx_buffer_size          = BUF_SAMPLES
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = rx_gain_db
    print(f"[PLUTO-RX] {CENTER_FREQ/1e6:.3f} MHz  SR={SAMPLE_RATE/1e6:.2f} MSPS  "
          f"Gain={rx_gain_db} dB")
    print("[PLUTO-RX] Receiving...  (Ctrl-C to stop)")

    _cb_count    = [0]
    _frame_count = [0]
    _orig_cb     = on_frame

    def _on_frame_counted(d):
        _frame_count[0] += 1
        if _orig_cb:
            _orig_cb(d)

    pipe = RX4FSKPipeline(_on_frame_counted)

    try:
        while True:
            raw = sdr.rx()
            # Pluto returns complex64 with raw ADC counts; normalise to ±1
            iq = np.asarray(raw, dtype=np.complex64) / 2048.0
            pipe.push_iq(iq)
            _cb_count[0] += 1
            if _cb_count[0] % 8 == 0:
                iq_rms = float(np.sqrt(np.mean(np.abs(iq)**2)))
                # IQ_RMS target: 0.05–0.40 (after /2048 normalisation).
                # < 0.02 → signal too weak (check gain, antenna, TX power).
                # > 0.50 → likely ADC saturation (reduce rx_gain_db).
                if iq_rms < 0.02:
                    warn = "  ⚠ WEAK SIGNAL — check TX power / antenna / gain"
                elif iq_rms > 0.50:
                    warn = "  ⚠ ADC SATURATION — reduce rx_gain_db"
                else:
                    warn = ""
                # Quick FM demod on raw IQ — carrier offset indicator
                # fm_mean ≈ 0 kHz   : no carrier offset  (✓ good)
                # fm_mean >> ±83 kHz : offset → slicing errors
                # fm_std  ≈ 100–300 kHz: consistent with 4-FSK signal
                # fm_std  <  30 kHz : CW or noise (no 4-FSK modulation)
                _fm = np.angle(iq[1:] * np.conj(iq[:-1])) * (SAMPLE_RATE / (2 * np.pi))
                fm_mean_khz = float(_fm.mean()) / 1e3
                fm_std_khz  = float(_fm.std())  / 1e3
                print(f"[DBG cb={_cb_count[0]:3d}]  "
                      f"IQ_RMS={iq_rms:.4f}  frames={_frame_count[0]}{warn}  "
                      f"FM: μ={fm_mean_khz:+.1f}kHz σ={fm_std_khz:.0f}kHz",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        print("\n[PLUTO-RX] Stopped.")


# ═══════════════════════════════════════════════════════════════
# 13b. Pluto RX Diagnostic Mode  (输出 DSP 链内部详细状态)
# ═══════════════════════════════════════════════════════════════
def run_pluto_diagnose(rx_gain_db: int = RX_GAIN_DB,
                      rx_uri: str = PLUTO_RX_URI,
                      n_blocks: int = 24):
    """
    Capture n_blocks of IQ and print a complete DSP-chain diagnostic.
    Run with: python rx_pluto_pipeline.py --hw --diagnose [--gain N]

    Interpretation guide
    --------------------
    FM μ ≈ 0 kHz          : no carrier offset  ✓
    FM μ >> ±83 kHz      : Pluto LO frequency error (slicing errors!)
    FM σ ≈ 100–280 kHz   : 4-FSK modulation detected  ✓
    FM σ < 30 kHz        : CW carrier or noise — no 4-FSK present!
    Symbol peaks at ±1,±3: demodulation OK; if frames=0 → CRC/timing bug
    No symbol peaks       : GardnerTED not locking
    """
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed")
        return

    _SEP = "\u2550" * 62
    print(f"\n{_SEP}")
    print(f"  DIAGNOSE  {CENTER_FREQ/1e6:.3f} MHz  SR={SAMPLE_RATE/1e6:.2f} MSPS  "
          f"gain={rx_gain_db} dB  n={n_blocks} blocks")
    print(_SEP)

    sdr = adi.Pluto(rx_uri)
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_hardwaregain_chan0   = -89
    sdr.rx_lo                  = int(CENTER_FREQ)
    sdr.sample_rate            = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth        = int(SAMPLE_RATE)
    sdr.rx_buffer_size         = BUF_SAMPLES
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = rx_gain_db

    lpf = PreFilter()
    fmd = FMDemod()
    mf  = MatchedFilter(RRC_KERNEL)
    agc = AGC()
    ted = GardnerTED(SPS, initial_offset=TOTAL_CHAIN_DELAY)

    iq_rms_list, freq_list, sym_list = [], [], []

    dur_ms = n_blocks * BUF_SAMPLES / SAMPLE_RATE * 1000
    print(f"Capturing {n_blocks} blocks ≈ {dur_ms:.0f} ms ...", end="", flush=True)
    for _ in range(n_blocks):
        raw = sdr.rx()
        iq  = np.asarray(raw, dtype=np.complex64) / 2048.0
        iq_rms_list.append(float(np.sqrt(np.mean(np.abs(iq)**2))))
        iq_filt   = lpf.run(iq)
        freq_norm = fmd.demod(iq_filt)
        mf_out    = mf.run(freq_norm)
        mf_agc    = agc.process(mf_out)
        ted_res   = ted.process(mf_agc)
        freq_list.extend(freq_norm.tolist())
        for sv, _ in ted_res:
            sym_list.append(sv)
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass
    print(" done")

    rms_a  = np.asarray(iq_rms_list)
    freq_a = np.asarray(freq_list, dtype=np.float32)
    sym_a  = np.asarray(sym_list,  dtype=np.float32)

    def _hist(arr, lo=-5.0, hi=5.0, n=20):
        cnt, edges = np.histogram(arr, bins=n, range=(lo, hi))
        pk = cnt.max() or 1
        lines = []
        for i, c in enumerate(cnt):
            center = (edges[i] + edges[i+1]) / 2
            bar = '\u2588' * (c * 38 // pk) if c > 0 else ''
            lines.append(f"  {center:+5.1f} \u2502 {bar} ({c})")
        return "\n".join(lines)

    # ── IQ RMS ───────────────────────────────────────────────────────────
    print(f"\n\u2500\u2500 IQ RMS (应在 0.05–0.40) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  mean={rms_a.mean():.4f}  std={rms_a.std():.5f}  "
          f"min={rms_a.min():.4f}  max={rms_a.max():.4f}")

    # ── FM Demod ─────────────────────────────────────────────────────────
    off_hz     = float(freq_a.mean()) * (FSK_DEVIATION / 3.0)
    std_norm   = float(freq_a.std())
    print(f"\n\u2500\u2500 FM Demod 输出 (归一化，4-FSK 理想峰在 \u00b11, \u00b13) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  \u03bc={freq_a.mean():+.4f} (≈ {off_hz/1e3:+.1f} kHz 载波偏移)  \u03c3={std_norm:.4f}")
    print(_hist(freq_a))

    # ── Symbols ──────────────────────────────────────────────────────────
    print(f"\n\u2500\u2500 MF + AGC + TED 输出符号采样 ({len(sym_a)} 个) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    if len(sym_a) > 0:
        print(f"  \u03bc={sym_a.mean():+.4f}  \u03c3={sym_a.std():.4f}")
        print(_hist(sym_a))
    else:
        print("  [!] 0 个符号 — TOTAL_CHAIN_DELAY 设置过大？来不及进行采样")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\u2500\u2500 诊断总结 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    if rms_a.mean() < 0.02:
        print("  \u26a0 无信号：IQ_RMS 过小 — 检查发射机功率/天线")
    elif rms_a.mean() > 0.50:
        print("  \u26a0 ADC 饶和：降低 rx_gain_db")
    else:
        print(f"  \u2713 IQ 电平正常 ({rms_a.mean():.3f})")

    if abs(freq_a.mean()) > 0.5:
        print(f"  \u26a0 载波偏移: {off_hz/1e3:+.1f} kHz \u2014 当 |offset| > 83 kHz 时切片错误")
        print(f"     建议: 检查 Pluto LO 校准或启用 AFC")
    else:
        print(f"  \u2713 载波偏移 OK ({off_hz/1e3:+.1f} kHz)")

    if std_norm < 0.3:
        print(f"  \u26a0 FM \u65b9差很小 (\u03c3={std_norm:.3f}) — 当前信号可能是 CW/噪声，不是 4-FSK!")
        print(f"     建议: 确认对方发射机正在运行并对准到此频率")
    elif std_norm < 2.5:
        print(f"  \u2713 FM 方差符合 4-FSK 预期 (\u03c3={std_norm:.3f})")
    else:
        print(f"  \u26a0 FM 方差过大 (\u03c3={std_norm:.3f}) — ADC 饶和或宽带噪声")

    if len(sym_a) == 0:
        print("  \u26a0 TED 返回 0 个符号 — 检查 TOTAL_CHAIN_DELAY")
    print()


# ═══════════════════════════════════════════════════════════════
# 13. Software Loopback Test
# ═══════════════════════════════════════════════════════════════
def _tx_4fsk_sim(symbols: np.ndarray) -> np.ndarray:
    upsampled = np.zeros(len(symbols) * SPS)
    upsampled[::SPS] = symbols
    freq_pulse = np.convolve(upsampled, RRC_KERNEL, mode='full')
    delay      = len(RRC_KERNEL) // 2
    freq_pulse = freq_pulse[delay: delay + len(upsampled)]
    freq_hz    = freq_pulse * (FSK_DEVIATION / 3.0)
    phase      = np.cumsum(freq_hz) * 2 * np.pi / SAMPLE_RATE
    return np.exp(1j * phase).astype(np.complex64) * 0.5


def software_loopback_test(key: str = "RM2026"):
    from fsk_digital_twin import (build_protocol_frame, bytes_to_bits, bits_to_symbols)

    print("=" * 58)
    print("Software Loopback Test — freq_offset=+5000 Hz, SNR=15 dB")
    print(f"  SR={SAMPLE_RATE/1e6:.2f} MSPS  SPS={SPS}  CENTER={CENTER_FREQ/1e6:.3f} MHz")
    print(f"  KEY='{key}'")
    print("=" * 58)

    frame = build_protocol_frame(key)
    bits  = bytes_to_bits(frame)
    syms  = bits_to_symbols(bits)
    print(f"TX Frame ({len(frame)}B): {frame.hex(' ').upper()}")

    REPS     = 5000
    syms_rep = np.tile(syms, REPS)
    tx_iq    = _tx_4fsk_sim(syms_rep)

    freq_offset_hz = 5000
    snr_db         = 15.0
    sig_power      = float(np.mean(np.abs(tx_iq)**2))
    noise_pwr      = sig_power / (10 ** (snr_db / 10))
    rng            = np.random.default_rng(42)
    noise = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(len(tx_iq)) + 1j * rng.standard_normal(len(tx_iq))
    )
    t_arr  = np.arange(len(tx_iq), dtype=np.float64) / SAMPLE_RATE
    rx_iq  = (tx_iq * np.exp(1j * 2 * np.pi * freq_offset_hz * t_arr)
              + noise).astype(np.complex64)

    received = []

    def record(d):
        received.append(d)
        print(f"  -> Decoded: {d}")

    pipe    = RX4FSKPipeline(on_frame=record)
    CHUNK   = BUF_SAMPLES
    n_chunks = 0
    for start in range(0, len(rx_iq), CHUNK):
        pipe.push_iq(rx_iq[start: start + CHUNK])
        n_chunks += 1

    print(f"\n  Chunks: {n_chunks}  ({n_chunks*CHUNK/SAMPLE_RATE*1000:.0f} ms total)")
    ok = any(r.get('key') == key for r in received)
    print(f"[{'PASS' if ok else 'FAIL'}] {len(received)} frame(s) decoded.")
    return received


# ─────────────────────────── Entry Point ───────────────────────
if __name__ == '__main__':
    """
    Usage:
      python rx_pluto_pipeline.py                      # software loopback self-test (key=RM2026)
      python rx_pluto_pipeline.py --key MyKey          # software loopback with custom key
      python rx_pluto_pipeline.py --hw                 # Pluto TX + Pluto RX
      python rx_pluto_pipeline.py --hw --rx-only
      python rx_pluto_pipeline.py --hw --tx-only
      python rx_pluto_pipeline.py --hw --key RM2026
      python rx_pluto_pipeline.py --hw --gain 40
      python rx_pluto_pipeline.py --hw --diagnose      # DSP chain diagnostic histogram
      python rx_pluto_pipeline.py --hw --diagnose --gain 45
    """
    hw      = '--hw'      in sys.argv
    rx_only = '--rx-only' in sys.argv
    tx_only = '--tx-only' in sys.argv

    key = 'RM2026'
    if '--key' in sys.argv:
        idx = sys.argv.index('--key')
        if idx + 1 < len(sys.argv):
            key = sys.argv[idx + 1]

    gain = RX_GAIN_DB
    if '--gain' in sys.argv:
        idx = sys.argv.index('--gain')
        if idx + 1 < len(sys.argv):
            gain = int(sys.argv[idx + 1])

    if hw and '--diagnose' in sys.argv:
        run_pluto_diagnose(rx_gain_db=gain)
        sys.exit(0)

    if not hw:
        software_loopback_test(key=key)
        sys.exit(0)

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
            run_pluto_rx(on_frame=_on_frame, rx_gain_db=gain)
        finally:
            if pluto:
                pluto.stop()
    else:
        print("[TX-ONLY] Pluto broadcasting. Press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if pluto:
                pluto.stop()
