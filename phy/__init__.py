"""
phy — RM2026 Physical Layer Modules
====================================
2-GFSK modem, air-packet deframer, payload stream reassembler,
clock recovery, filter design, and legacy 4-RRC-FSK path.

Modules are loaded lazily as they are implemented across phases.
"""

# Phase 1 — always available
from phy.air_packet import AirPacketDeframer, ac_to_bits
from phy.stream_reassembler import PayloadStreamReassembler

# Phase 2 — filters + clock recovery
try:
    from phy.filters import make_gaussian_filter, make_rrc
except ImportError:
    pass

try:
    from phy.clock_recovery import BlockPhaseClockRecovery
except ImportError:
    pass

# Phase 3 — 2-GFSK modem
try:
    from phy.gfsk2_modem import GFSK2Demodulator, gfsk2_modulate_bits
except ImportError:
    pass

# Phase 4 — legacy 4-RRC-FSK
try:
    from phy.legacy_4rrcfsk import Legacy4RRCFSKModem
except ImportError:
    pass

__all__ = [
    "AirPacketDeframer",
    "ac_to_bits",
    "PayloadStreamReassembler",
    "make_gaussian_filter",
    "make_rrc",
    "BlockPhaseClockRecovery",
    "GFSK2Demodulator",
    "gfsk2_modulate_bits",
    "Legacy4RRCFSKModem",
]
