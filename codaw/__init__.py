"""codaw — a codified digital audio workstation.

Everything is code: oscillators, envelopes, instruments, effects, tracks and a
mixer are all composable Python modules, and a song is a script that constructs
a :class:`Song` and renders or plays it.

This module re-exports the curated, fully-typed public API. The surface grows
as features land; for the MVP it exposes :class:`Signal`, the ``oscillators``
module and :class:`Song`.
"""

from __future__ import annotations

from codaw.arrangement.song import Song
from codaw.core import oscillators
from codaw.core.signal import Signal

__version__ = "0.0.1"

__all__ = [
    "Signal",
    "Song",
    "oscillators",
    "__version__",
]
