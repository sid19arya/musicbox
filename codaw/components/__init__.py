"""Layer 2 — modular DAW components: instruments, effects and note events."""

from __future__ import annotations

from codaw.components.events import NoteEvent
from codaw.components.instrument import Instrument, Synth

__all__ = ["Instrument", "Synth", "NoteEvent"]
