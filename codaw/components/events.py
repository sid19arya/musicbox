"""Note events: a pitch placed in musical time with a duration and velocity."""

from __future__ import annotations

from dataclasses import dataclass

from codaw.music.pitch import Note

__all__ = ["NoteEvent"]


@dataclass(frozen=True)
class NoteEvent:
    """A single note placed on a timeline, measured in beats.

    Parameters
    ----------
    pitch:
        The note's pitch.
    start_beat:
        Onset position in beats from the start of the pattern.
    duration_beats:
        Sounding length in beats.
    velocity:
        Loudness in ``[0, 1]`` (a linear amplitude scale).
    """

    pitch: Note
    start_beat: float
    duration_beats: float
    velocity: float = 1.0

    # Allow constructing from a name string or MIDI int for ergonomics.
    def __post_init__(self) -> None:
        if not isinstance(self.pitch, Note):
            object.__setattr__(self, "pitch", Note.coerce(self.pitch))
        if self.start_beat < 0:
            raise ValueError(f"start_beat must be non-negative, got {self.start_beat}")
        if self.duration_beats <= 0:
            raise ValueError(f"duration_beats must be positive, got {self.duration_beats}")
        if not 0.0 <= self.velocity <= 1.0:
            raise ValueError(f"velocity must be in [0, 1], got {self.velocity}")

    @property
    def end_beat(self) -> float:
        """Beat position at which the note ends."""
        return self.start_beat + self.duration_beats
