"""Layer 1 — music theory: notes, scales/chords, and timing."""

from __future__ import annotations

from codaw.music import timing
from codaw.music.pitch import Note
from codaw.music.scales import Scale, chord
from codaw.music.timing import Duration, beats_to_seconds, seconds_to_beats

__all__ = [
    "Note",
    "Scale",
    "chord",
    "Duration",
    "beats_to_seconds",
    "seconds_to_beats",
    "timing",
]
