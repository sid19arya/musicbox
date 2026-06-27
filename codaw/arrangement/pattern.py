"""Patterns: ordered collections of :class:`NoteEvent` placed in beats."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from codaw.components.events import NoteEvent
from codaw.music.pitch import Note

__all__ = ["Pattern"]

# Tokens that denote a rest (advance time without sounding a note).
_REST_TOKENS = {"_", "r", "R", "."}


class Pattern:
    """An ordered collection of :class:`NoteEvent`s.

    Build patterns directly from events, sequentially from a compact string
    (:meth:`from_string`), or from a list of pitches (:meth:`sequence`).

    Parameters
    ----------
    events:
        Initial note events. Defaults to empty.
    """

    def __init__(self, events: Iterable[NoteEvent] | None = None) -> None:
        self.events: list[NoteEvent] = list(events) if events is not None else []

    def add(self, event: NoteEvent) -> Pattern:
        """Append a note event; returns ``self`` for chaining."""
        self.events.append(event)
        return self

    @property
    def length_beats(self) -> float:
        """The pattern's length in beats (the latest event end, or 0)."""
        return max((e.end_beat for e in self.events), default=0.0)

    @classmethod
    def from_string(cls, spec: str, *, default_velocity: float = 1.0) -> Pattern:
        """Parse a compact pattern string into sequential note events.

        Each whitespace-separated token is ``NOTE:DURATION`` where ``NOTE`` is a
        note name (e.g. ``C4``) or a rest (``_``, ``R`` or ``.``) and
        ``DURATION`` is in beats. An optional ``@VELOCITY`` suffix sets the
        velocity, e.g. ``"C4:1@0.8"``. Notes are laid end to end.

        Examples
        --------
        ``Pattern.from_string("C4:1 E4:1 G4:2")`` → three notes at beats 0, 1, 2.
        """
        events: list[NoteEvent] = []
        beat = 0.0
        for token in spec.split():
            note_part, sep, rest = token.partition(":")
            if not sep:
                raise ValueError(f"invalid token {token!r}: expected NOTE:DURATION")
            dur_part, _, vel_part = rest.partition("@")
            try:
                duration = float(dur_part)
                velocity = float(vel_part) if vel_part else default_velocity
            except ValueError as exc:
                raise ValueError(f"invalid token {token!r}: {exc}") from exc
            if note_part in _REST_TOKENS:
                beat += duration
                continue
            events.append(NoteEvent(Note.from_name(note_part), beat, duration, velocity))
            beat += duration
        return cls(events)

    @classmethod
    def sequence(
        cls,
        notes: Sequence[Note | str | int],
        dur: float = 0.5,
        *,
        velocity: float = 1.0,
        start_beat: float = 0.0,
    ) -> Pattern:
        """Build a pattern of equal-duration notes laid end to end.

        Parameters
        ----------
        notes:
            Pitches as :class:`Note`, names or MIDI numbers.
        dur:
            Duration of each note in beats.
        velocity:
            Velocity applied to every note.
        start_beat:
            Beat at which the sequence begins.
        """
        events: list[NoteEvent] = []
        beat = start_beat
        for n in notes:
            events.append(NoteEvent(Note.coerce(n), beat, dur, velocity))
            beat += dur
        return cls(events)

    def __iter__(self) -> Iterator[NoteEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"Pattern({len(self.events)} events, {self.length_beats} beats)"
