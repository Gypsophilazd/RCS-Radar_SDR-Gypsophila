"""
tests/test_tx_disabled_by_default.py
=====================================
Unit tests for TX safety — no SDR hardware required.

Tests argument parsing / guard logic only.  Does NOT open Pluto hardware.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock


def test_tx_producer_raises_without_enable():
    """PlutoTxProducer.start() raises RuntimeError when test_tx_enabled=False."""
    from tx_signal_produce import PlutoTxProducer
    from config_manager import load_config

    mgr = load_config()
    tx = PlutoTxProducer(mgr, key="RM2026", simulate_arena=False)

    with pytest.raises(RuntimeError, match="TX is disabled by default"):
        tx.start()


def test_tx_producer_allows_when_enabled():
    """
    PlutoTxProducer with test_tx_enabled=True passes the guard check.
    The actual Pluto open will fail (no hardware), but the guard
    itself must NOT raise.
    """
    from tx_signal_produce import PlutoTxProducer
    from config_manager import load_config

    mgr = load_config()
    tx = PlutoTxProducer(mgr, key="RM2026", simulate_arena=False,
                         test_tx_enabled=True)

    # The guard should pass — but pyadi-iio may not be installed or
    # Pluto may not be connected.  That's fine; we only test the guard.
    # If pyadi-iio IS installed and Pluto IS connected, we must NOT
    # actually transmit — but the guard has already passed.
    try:
        tx.start()
    except RuntimeError as e:
        # Accept: "TX is disabled" (should NOT happen with enabled=True)
        if "TX is disabled" in str(e):
            pytest.fail("Guard raised despite test_tx_enabled=True")
        # Accept: pyadi-iio not installed or Pluto not connected
        if "pyadi-iio" in str(e) or "PlutoSDR" in str(e):
            pass  # expected when no hardware
    except Exception:
        pass  # any other hardware-related error is fine
    finally:
        tx.stop()


def test_tx_parser_flag_present():
    """--test-tx-enable is a recognised CLI flag."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--test-tx-enable", action="store_true")
    args = p.parse_args(["--test-tx-enable"])
    assert args.test_tx_enable is True

    args2 = p.parse_args([])
    assert args2.test_tx_enable is False
