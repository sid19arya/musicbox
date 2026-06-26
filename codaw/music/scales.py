"""Scales and chords built from :class:`Note` intervals.

A :class:`Scale` yields the notes of a named scale from a root; :func:`chord`
parses common chord symbols (e.g. ``"Cmaj7"``, ``"Am"``, ``"G7"``) into notes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from codaw.music.pitch import Note

__all__ = ["Scale", "chord", "SCALE_INTERVALS", "CHORD_INTERVALS"]

# Semitone intervals from the root for one octave of each scale.
SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "chromatic": tuple(range(12)),
}

# Semitone intervals from the root for each chord quality.
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "maj": (0, 4, 7),
    "M": (0, 4, 7),
    "m": (0, 3, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "halfdim": (0, 3, 6, 10),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
}

_CHORD_RE = re.compile(r"^([A-Ga-g][#♯b♭]*)(.*)$")


class Scale:
    """A musical scale: a root note plus a named interval set.

    Parameters
    ----------
    root:
        Root pitch as a :class:`Note`, a name string (``"C"`` defaults to
        octave 4, or ``"C3"``), or a MIDI int.
    name:
        Scale name, e.g. ``"major"``, ``"minor"``, ``"dorian"``,
        ``"minor_pentatonic"`` (see :data:`SCALE_INTERVALS`).
    """

    def __init__(self, root: Note | str | int, name: str = "major") -> None:
        self.root = _coerce_root(root)
        key = name.lower()
        if key not in SCALE_INTERVALS:
            raise ValueError(f"unknown scale {name!r}; known scales: {sorted(SCALE_INTERVALS)}")
        self.name = key
        self.intervals = SCALE_INTERVALS[key]

    def notes(self, octaves: int = 1) -> list[Note]:
        """Return the scale's notes spanning ``octaves`` octaves from the root.

        The octave root is included at the top (e.g. a 1-octave major scale has
        8 notes, root..root+12).
        """
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")
        result: list[Note] = []
        for o in range(octaves):
            for interval in self.intervals:
                result.append(self.root.transpose(o * 12 + interval))
        result.append(self.root.transpose(octaves * 12))
        return result

    def degree(self, n: int) -> Note:
        """Return the note at scale degree ``n`` (1-based), wrapping octaves."""
        if n < 1:
            raise ValueError(f"degree is 1-based, got {n}")
        size = len(self.intervals)
        octave, idx = divmod(n - 1, size)
        return self.root.transpose(octave * 12 + self.intervals[idx])

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes())

    def __repr__(self) -> str:
        return f"Scale(root={self.root.name!r}, name={self.name!r})"


def chord(symbol: str, octave: int = 4) -> list[Note]:
    """Parse a chord symbol into its notes.

    Parameters
    ----------
    symbol:
        A chord symbol such as ``"C"``, ``"Am"``, ``"G7"``, ``"Cmaj7"``,
        ``"F#m7b5"``. A bare letter is a major triad. An explicit octave in the
        root (``"C3maj7"``) overrides the ``octave`` argument.
    octave:
        Default octave for the root when the symbol omits one.

    Returns
    -------
    list[Note]
        The chord's notes, lowest first.
    """
    match = _CHORD_RE.match(symbol.strip())
    if match is None:
        raise ValueError(f"invalid chord symbol: {symbol!r}")
    root_str, quality = match.groups()
    # Qualities like "7"/"9"/"6" start with digits, so only treat a leading
    # number as an embedded octave (e.g. "C3maj7") when the remainder is itself
    # a valid quality and the whole string is not.
    if quality not in CHORD_INTERVALS:
        oct_match = re.match(r"^(-?\d+)(.*)$", quality)
        if oct_match and oct_match.group(2) in CHORD_INTERVALS:
            octave = int(oct_match.group(1))
            quality = oct_match.group(2)
    if quality not in CHORD_INTERVALS:
        raise ValueError(f"unknown chord quality {quality!r} in {symbol!r}")
    root = Note.from_name(f"{root_str}{octave}")
    return [root.transpose(i) for i in CHORD_INTERVALS[quality]]


def _coerce_root(root: Note | str | int) -> Note:
    """Coerce a scale root; bare letter names default to octave 4."""
    if isinstance(root, str) and re.fullmatch(r"[A-Ga-g][#♯b♭]*", root.strip()):
        return Note.from_name(f"{root.strip()}4")
    return Note.coerce(root)
