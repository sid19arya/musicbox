"""Layer 0 — DSP primitives: :class:`Signal`, oscillators, and (later) envelopes,
DSP helpers and filters."""

from __future__ import annotations

from codaw.core import oscillators
from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal

__all__ = ["ADSR", "Signal", "oscillators"]
