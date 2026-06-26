"""Timing: converting between musical beats and seconds, plus note-value constants.

Tempo is given in beats per minute (BPM), where one beat is a quarter note.
"""

from __future__ import annotations

__all__ = [
    "beats_to_seconds",
    "seconds_to_beats",
    "Duration",
    "WHOLE",
    "HALF",
    "QUARTER",
    "EIGHTH",
    "SIXTEENTH",
    "THIRTYSECOND",
    "DOTTED_HALF",
    "DOTTED_QUARTER",
    "DOTTED_EIGHTH",
    "TRIPLET_QUARTER",
    "TRIPLET_EIGHTH",
]


def beats_to_seconds(beats: float, bpm: float) -> float:
    """Convert a duration in beats (quarter notes) to seconds at ``bpm``."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if beats < 0:
        raise ValueError(f"beats must be non-negative, got {beats}")
    return beats * 60.0 / bpm


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """Convert a duration in seconds to beats (quarter notes) at ``bpm``."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    return seconds * bpm / 60.0


class Duration:
    """A musical duration measured in beats (quarter-note units).

    Wraps a beat count so it can be converted to seconds for a given tempo and
    combined arithmetically.

    Parameters
    ----------
    beats:
        Length in beats (a quarter note is 1.0 beat).
    """

    def __init__(self, beats: float) -> None:
        if beats < 0:
            raise ValueError(f"beats must be non-negative, got {beats}")
        self.beats = float(beats)

    def seconds(self, bpm: float) -> float:
        """Return this duration in seconds at ``bpm``."""
        return beats_to_seconds(self.beats, bpm)

    def dotted(self) -> Duration:
        """Return the dotted form (1.5x) of this duration."""
        return Duration(self.beats * 1.5)

    def triplet(self) -> Duration:
        """Return the triplet form (2/3x) of this duration."""
        return Duration(self.beats * 2.0 / 3.0)

    def __add__(self, other: Duration) -> Duration:
        if not isinstance(other, Duration):
            return NotImplemented
        return Duration(self.beats + other.beats)

    def __mul__(self, factor: float) -> Duration:
        if not isinstance(factor, (int, float)):
            return NotImplemented
        return Duration(self.beats * factor)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Duration):
            return NotImplemented
        return self.beats == other.beats

    def __hash__(self) -> int:
        return hash(self.beats)

    def __repr__(self) -> str:
        return f"Duration(beats={self.beats})"


# Note-value constants, in beats (quarter note = 1 beat).
WHOLE = 4.0
HALF = 2.0
QUARTER = 1.0
EIGHTH = 0.5
SIXTEENTH = 0.25
THIRTYSECOND = 0.125

DOTTED_HALF = HALF * 1.5
DOTTED_QUARTER = QUARTER * 1.5
DOTTED_EIGHTH = EIGHTH * 1.5

TRIPLET_QUARTER = QUARTER * 2.0 / 3.0
TRIPLET_EIGHTH = EIGHTH * 2.0 / 3.0
