"""
phy/stream_reassembler.py
=========================
PayloadStreamReassembler — reassembles 15-byte air-packet payload chunks
into complete RoboMaster frames.

The reassembler accumulates payload bytes into a continuous stream buffer,
searches for SOF 0xA5, and extracts full RM frames (validated by CRC8 header
and CRC16 frame check). Frames may span multiple 15-byte payload chunks.

Returns raw RM frame bytes only — decoding is the caller's responsibility
(DSPProcessor calls packet_decoder.decode_frame()).

Usage::

    reassembler = PayloadStreamReassembler(max_buffer=4096)
    raw_frames = reassembler.push_payload(payload)   # list[bytes]
"""

from __future__ import annotations

import struct
from typing import List

# ── CRC imports from packet_decoder (public exports) ────────────────────────
from packet_decoder import crc8_rm, crc16_rm

SOF = 0xA5
_MIN_FRAME_LEN = 9   # SOF(1)+DataLen(2)+Seq(1)+CRC8(1)+CmdID(2)+CRC16(2)
_MAX_DATA_LEN  = 64  # sanity guard against corrupted DataLen fields


class PayloadStreamReassembler:
    """
    Accumulates 15-byte payload chunks and yields complete RM frames.

    Parameters
    ----------
    max_buffer : int
        Maximum stream buffer size in bytes.  When exceeded the oldest
        data is discarded to prevent unbounded memory growth.
    """

    def __init__(self, max_buffer: int = 4096):
        self._max_buf = max_buffer
        self._stream = bytearray()

    # ── public ──────────────────────────────────────────────────────────────

    def push_payload(self, payload: bytes) -> list[bytes]:
        """
        Append a 15-byte air-packet payload and return any complete RM frames.

        Parameters
        ----------
        payload : bytes (typically 15 bytes)

        Returns
        -------
        list[bytes] — zero or more raw RM frame byte strings
        """
        self._stream.extend(payload)

        # Buffer overflow protection: discard oldest bytes
        if len(self._stream) > self._max_buf:
            self._stream = self._stream[-self._max_buf:]

        frames: list[bytes] = []
        while True:
            frame = self._try_extract_frame()
            if frame is None:
                break
            frames.append(frame)
        return frames

    # ── private ─────────────────────────────────────────────────────────────

    def _try_extract_frame(self) -> bytes | None:
        """
        Try to extract one complete RM frame from the stream buffer.

        Returns the raw frame bytes if a valid frame is found and its CRC
        passes, otherwise None.  The extracted bytes are removed from the
        stream on success.
        """
        # Find SOF
        sof_idx = self._stream.find(bytes([SOF]))
        if sof_idx < 0:
            # No SOF — discard everything (noise)
            self._stream.clear()
            return None

        # Discard bytes before SOF (noise / partial previous frame)
        if sof_idx > 0:
            del self._stream[:sof_idx]

        # Need at least header + CRC8 to determine frame length
        if len(self._stream) < 5:
            return None

        # Verify CRC8 over header (bytes 0-3, CRC8 at byte 4)
        if crc8_rm(bytes(self._stream[:4])) != self._stream[4]:
            # Bad header — discard SOF byte and try again
            del self._stream[:1]
            return None

        # Parse DataLen (little-endian uint16 at bytes 1-2)
        data_len = struct.unpack_from("<H", self._stream, 1)[0]
        if data_len < 2 or data_len > _MAX_DATA_LEN:
            del self._stream[:1]
            return None

        # Total frame = header(4) + CRC8(1) + DataLen + CRC16(2)
        total_frame = 5 + data_len + 2
        if len(self._stream) < total_frame:
            return None   # wait for more data

        # Verify CRC16 over all bytes except the last 2
        frame_bytes = bytes(self._stream[:total_frame])
        expected_crc16 = struct.unpack_from("<H", frame_bytes, total_frame - 2)[0]
        if crc16_rm(frame_bytes[:-2]) != expected_crc16:
            # CRC16 failed — discard SOF and hunt again
            del self._stream[:1]
            return None

        # Success: remove frame from stream and return
        del self._stream[:total_frame]
        return frame_bytes
