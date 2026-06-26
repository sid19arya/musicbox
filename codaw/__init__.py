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
from codaw.components.events import NoteEvent
from codaw.components.instrument import Instrument, Synth
from codaw.core import oscillators
from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal
from codaw.music.pitch import Note
from codaw.music.scales import Scale, chord

__version__ = "0.0.1"

__all__ = [
    "ADSR",
    "Instrument",
    "Note",
    "NoteEvent",
    "Scale",
    "Signal",
    "Song",
    "Synth",
    "chord",
    "oscillators",
    "__version__",
]
