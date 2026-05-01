"""
dsp_processor.py
================
Module 4a — DSP Processor
RCS-Radar-SDR  RM2026 Arena System

Streaming demodulation pipeline with mode dispatch:

  Mode "2gfsk" (default):
    IQ blocks → GFSK2Demodulator → AirPacketDeframer →
    PayloadStreamReassembler → decode_frame → callback

  Mode "4rrcfsk_legacy":
    IQ blocks → Legacy4RRCFSKModem → FrameSync → decode_frame → callback

All filters preserve cross-block state — critical for correct streaming
demodulation.

Threading model
───────────────
  DSPProcessor.run_forever() blocks the calling thread.
  Call it in a daemon thread from main.py.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config_manager import ConfigManager, FreqPlan

from packet_decoder import decode_frame, crc8_rm, verify_frame
from phy.gfsk2_modem import GFSK2Demodulator
from phy.air_packet import AirPacketDeframer
from phy.stream_reassembler import PayloadStreamReassembler
from phy.legacy_4rrcfsk import Legacy4RRCFSKModem
from phy.rf_profiler import RfProfiler

# ── Maximum plausible DataLen ────────────────────────────────────────────────
_MAX_DATA_LEN = 64   # CmdID(2) + max-payload(36) + CRC16(2) + margin


class DSPProcessor:
    """
    Consumes raw IQ blocks, demodulates, and calls *on_frame* for each
    decoded and CRC-verified RoboMaster frame.

    Parameters
    ----------
    config    : loaded ConfigManager
    in_queue  : queue.Queue[np.ndarray] fed by PlutoRxDriver (complex64)
    on_frame  : callback(dict) invoked in DSP thread for each decoded frame
    out_iq_q  : optional queue for raw IQ → GUI spectrum / waveform
    out_sym_q : optional queue for symbol values → GUI scatter panel
    rx_source : "broadcast" (default) or "jammer" — RX listen target
    direct_tune : bool
        If True, the SDR is directly tuned to the channel of interest
        at 1 MSPS — no channelizer/decimation needed regardless of
        plan.channelize.
    """

    def __init__(
        self,
        config: "ConfigManager",
        in_queue: "queue.Queue[np.ndarray]",
        on_frame: Callable[[dict], None],
        out_iq_q:  Optional["queue.Queue"] = None,
        out_sym_q: Optional["queue.Queue[np.ndarray]"] = None,
        rx_source: str = "broadcast",
        direct_tune: bool = False,
    ):
        self._cfg      = config
        self._q_in     = in_queue
        self._on_frame = on_frame
        self._q_iq     = out_iq_q
        self._q_sym    = out_sym_q
        self._rx_source = rx_source

        plan          = config.plan
        phy           = config.phy_config
        self._sr      = plan.sample_rate_hz
        self._centre_hz = float(plan.center_freq_hz)
        self._phy_mode = phy.mode

        # ── Build modem chain by mode ───────────────────────────────────────
        if phy.mode == "2gfsk":
            # Select deviation and channelizer offset based on RX source
            # When direct_tune is True, the SDR is already tuned to the channel
            # at 1 MSPS — no channelization/decimation needed.
            if rx_source == "jammer":
                rx_dev = phy.jammer_deviation_hz
                chan_offset = (plan.jammer_offset_hz
                               if (plan.channelize and not direct_tune
                                   and plan.jammer_offset_hz != 0.0)
                               else None)
                ac_mode = "jammer"
            else:
                rx_dev = phy.deviation_hz
                chan_offset = (plan.broadcast_offset_hz
                               if (plan.channelize and not direct_tune)
                               else None)
                ac_mode = phy.access_code_mode

            # 2-GFSK chain: demod → deframer → reassembler → decode
            self._demod = GFSK2Demodulator(
                sps=phy.sps,
                bt=phy.bt,
                span=phy.span,
                deviation_hz=rx_dev,
                sample_rate=phy.sample_rate,
                input_sample_rate=float(plan.sample_rate_hz),
                channelizer_offset_hz=chan_offset,
                threshold_mode="zero",
                sub_block_syms=phy.sub_block_syms,
            )
            self._deframer = AirPacketDeframer(mode=ac_mode)
            self._reassembler = PayloadStreamReassembler()
            self._legacy_modem = None

            # ── Frame-sync state machine (unused in 2-GFSK mode) ────────────
            self._bit_sreg    : int       = 0
            self._frame_state : str       = "HUNT"
            self._frame_buf   : list[int] = []
            self._expected_body: int      = 0
            self._header_bits : int       = 0
            self._body_bits   : int       = 0

        elif phy.mode == "4rrcfsk_legacy":
            self._legacy_modem = Legacy4RRCFSKModem(
                sample_rate=self._sr,
                baud_rate=250_000,
                channelize=plan.channelize,
                broadcast_offset_hz=plan.broadcast_offset_hz,
            )
            self._demod = None
            self._deframer = None
            self._reassembler = None

            # ── Frame-sync state machine (used in legacy mode) ──────────────
            self._bit_sreg    : int       = 0
            self._frame_state : str       = "HUNT"
            self._frame_buf   : list[int] = []
            self._expected_body: int      = 0
            self._header_bits : int       = 0
            self._body_bits   : int       = 0
        else:
            raise ValueError(f"Unknown phy_mode: {phy.mode!r}")

        # ── RF profiler (semi_auto gain support) ─────────────────────────────
        self._profiler = RfProfiler(
            gain_db=config.rx_gain_db,
            gain_mode=config.gain_mode,
        )
        self._profiler_summary_blocks = 0

        self._stop_event = threading.Event()

    # ── public ───────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Block and process IQ blocks until stop() is called."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                iq = self._q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_block(iq)

    def stop(self) -> None:
        self._stop_event.set()

    # ── block pipeline ───────────────────────────────────────────────────────

    @property
    def profiler(self) -> RfProfiler:
        return self._profiler

    def _process_block(self, iq: np.ndarray) -> None:
        iq = np.asarray(iq, dtype=np.complex64)

        # Profiler: update IQ RMS stats, check for gain adjustments
        gain_delta = self._profiler.update(iq)

        # Forward raw IQ to GUI
        if self._q_iq is not None:
            try:
                self._q_iq.put_nowait((iq, self._centre_hz, self._sr))
            except queue.Full:
                pass

        if self._phy_mode == "2gfsk":
            self._process_block_2gfsk(iq)
        else:
            self._process_block_legacy(iq)

        # Print profiler summary every ~8 blocks (~2 s at 1 MSPS 262144-sample blocks)
        self._profiler_summary_blocks += 1
        if self._profiler_summary_blocks % 8 == 0:
            print(self._profiler.summary())
            self._profiler.reset_counts()

    # ── 2-GFSK path ──────────────────────────────────────────────────────────

    def _process_block_2gfsk(self, iq: np.ndarray) -> None:
        """2-GFSK: demodulate → deframe → reassemble → decode."""
        # 1. Demodulate IQ to bits
        bits = self._demod.push_iq(iq)

        # 2. Deframe: hunt access codes, extract 15-byte payloads
        payloads = self._deframer.push_bits(bits)

        # Track AC hits and payloads
        if payloads:
            self._profiler.record_ac_hit()
            self._profiler.record_payload()

        # 3. Reassemble payload stream into RM frames, decode, callback
        for payload in payloads:
            raw_frames = self._reassembler.push_payload(payload)
            for raw in raw_frames:
                self._profiler.record_raw_frame()
                crc_ok = verify_frame(raw)
                self._profiler.record_crc(crc_ok)
                if crc_ok:
                    result = decode_frame(raw)
                    if result is not None:
                        self._profiler.record_decoded_frame()
                        self._on_frame(result)
                else:
                    # Frame bytes exist but CRC failed
                    pass

    # ── Legacy 4-RRC-FSK path ────────────────────────────────────────────────

    def _process_block_legacy(self, iq: np.ndarray) -> None:
        """Legacy 4-RRC-FSK: demodulate → FrameSync → decode."""
        bits = self._legacy_modem.push_iq(iq)

        if self._q_sym is not None and len(bits):
            try:
                self._q_sym.put_nowait(np.array(bits[-128:], dtype=np.float32))
            except queue.Full:
                pass

        for bit in bits:
            self._push_bit(bit)

    def _on_legacy_frame(self, result: dict) -> None:
        """Wrapper for legacy frame callback — records profiling stats."""
        self._profiler.record_raw_frame()
        self._profiler.record_crc(True)
        self._profiler.record_decoded_frame()
        self._on_frame(result)

    # ── Frame-sync state machine (legacy mode) ───────────────────────────────

    def _push_bit(self, bit: int) -> None:
        """
        Feed one bit through the SOF-hunting state machine.

        States: HUNT → HEADER → BODY → decode → HUNT
        """
        self._bit_sreg = ((self._bit_sreg << 1) | (bit & 1)) & 0xFF

        if self._frame_state == "HUNT":
            if self._bit_sreg == 0xA5:
                self._frame_buf   = [0xA5]
                self._header_bits = 0
                self._frame_state = "HEADER"

        elif self._frame_state == "HEADER":
            self._header_bits += 1
            if self._header_bits % 8 == 0:
                self._frame_buf.append(self._bit_sreg)
            if self._header_bits == 32:
                if crc8_rm(bytes(self._frame_buf[:4])) != self._frame_buf[4]:
                    self._frame_state = "HUNT"
                    return
                data_len = self._frame_buf[1] | (self._frame_buf[2] << 8)
                if data_len == 0 or data_len > _MAX_DATA_LEN:
                    self._frame_state = "HUNT"
                    return
                self._expected_body = data_len + 2
                self._body_bits     = 0
                self._frame_state   = "BODY"

        elif self._frame_state == "BODY":
            self._body_bits += 1
            if self._body_bits % 8 == 0:
                self._frame_buf.append(self._bit_sreg)
            if self._body_bits == self._expected_body * 8:
                result = decode_frame(bytes(self._frame_buf))
                if result is not None:
                    self._on_legacy_frame(result)
                self._frame_state = "HUNT"
