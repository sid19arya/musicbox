"""Pitch: the :class:`Note` type converting between names, MIDI numbers and Hz.

Uses 12-tone equal temperament with A4 = 440 Hz (MIDI 69).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Note"]

# Semitone offset of each natural note from C within an octave.
_NATURAL_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"#": 1, "♯": 1, "b": -1, "♭": -1}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#♯b♭]*)(-?\d+)$")

# Canonical sharp spelling for each semitone class, for name round-tripping.
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

A4_MIDI = 69
A4_FREQ = 440.0


@dataclass(frozen=True)
class Note:
    """A musical pitch identified by its MIDI number.

    A :class:`Note` can be built from a name (``"C4"``, ``"A#3"``, ``"Bb5"``) or
    a MIDI number, and exposes its frequency in Hz. Middle C is ``"C4"`` (MIDI
    60); ``"A4"`` is MIDI 69 at 440 Hz.

    Parameters
    ----------
    midi:
        MIDI note number (0–127 in standard MIDI, though any int is accepted).
    """

    midi: int

    def __post_init__(self) -> None:
        if not isinstance(self.midi, int):
            raise TypeError(f"midi must be an int, got {type(self.midi).__name__}")

    @classmethod
    def from_name(cls, name: str) -> Note:
        """Parse a note name like ``"C4"``, ``"A#3"`` or ``"Bb5"``.

        The octave uses scientific pitch notation (C4 = middle C = MIDI 60).
        Multiple accidentals (e.g. ``"Cbb4"``) are summed.
        """
        match = _NOTE_RE.match(name.strip())
        if match is None:
            raise ValueError(f"invalid note name: {name!r}")
        letter, accidentals, octave = match.groups()
        semitone = _NATURAL_SEMITONES[letter.upper()]
        semitone += sum(_ACCIDENTALS[a] for a in accidentals)
        midi = (int(octave) + 1) * 12 + semitone
        return cls(midi)

    @classmethod
    def coerce(cls, value: Note | str | int) -> Note:
        """Coerce a :class:`Note`, a name string, or a MIDI int into a Note."""
        if isinstance(value, Note):
            return value
        if isinstance(value, str):
            return cls.from_name(value)
        if isinstance(value, int):
            return cls(value)
        raise TypeError(f"cannot coerce {type(value).__name__} to Note")

    @property
    def frequency(self) -> float:
        """Frequency in Hz (A4 = 440, 12-TET)."""
        return float(A4_FREQ * 2.0 ** ((self.midi - A4_MIDI) / 12.0))

    @property
    def name(self) -> str:
        """Canonical sharp-spelled name, e.g. ``"C#4"``."""
        octave = self.midi // 12 - 1
        return f"{_SHARP_NAMES[self.midi % 12]}{octave}"

    def transpose(self, semitones: int) -> Note:
        """Return a new note shifted by ``semitones`` (positive = up)."""
        return Note(self.midi + semitones)

    def __str__(self) -> str:
        return self.name
