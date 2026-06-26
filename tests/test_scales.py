"""Tests for scales and chords."""

from __future__ import annotations

import pytest

from codaw.music.pitch import Note
from codaw.music.scales import Scale, chord


def names(notes: list[Note]) -> list[str]:
    return [n.name for n in notes]


def test_c_major_scale() -> None:
    scale = Scale("C", "major")
    assert names(scale.notes()) == ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]


def test_a_minor_scale() -> None:
    scale = Scale("A", "minor")
    assert names(scale.notes()) == ["A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5"]


def test_pentatonic_has_five_plus_octave() -> None:
    scale = Scale("C", "minor_pentatonic")
    notes = scale.notes()
    assert len(notes) == 6  # 5 degrees + octave
    assert names(notes) == ["C4", "D#4", "F4", "G4", "A#4", "C5"]


def test_scale_iter() -> None:
    scale = Scale("C", "major")
    assert len(list(scale)) == 8


def test_scale_multiple_octaves() -> None:
    scale = Scale("C", "major")
    notes = scale.notes(octaves=2)
    assert len(notes) == 15  # 7*2 + 1
    assert notes[0].name == "C4"
    assert notes[-1].name == "C6"


def test_scale_degree_is_one_based_and_wraps() -> None:
    scale = Scale("C", "major")
    assert scale.degree(1).name == "C4"
    assert scale.degree(3).name == "E4"
    assert scale.degree(8).name == "C5"  # wraps to next octave


def test_scale_accepts_note_and_int_roots() -> None:
    assert Scale(Note(60), "major").root.name == "C4"
    assert Scale(60, "major").root.name == "C4"
    assert Scale("C3", "major").root.name == "C3"


def test_unknown_scale_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scale"):
        Scale("C", "bogus")


def test_scale_bad_octaves_and_degree() -> None:
    scale = Scale("C", "major")
    with pytest.raises(ValueError, match="octaves"):
        scale.notes(octaves=0)
    with pytest.raises(ValueError, match="1-based"):
        scale.degree(0)


def test_scale_repr() -> None:
    assert repr(Scale("C", "major")) == "Scale(root='C4', name='major')"


def test_chord_major_triad() -> None:
    assert names(chord("C")) == ["C4", "E4", "G4"]
    assert names(chord("Cmaj")) == ["C4", "E4", "G4"]


def test_chord_minor_and_sevenths() -> None:
    assert names(chord("Am")) == ["A4", "C5", "E5"]
    assert names(chord("Cmaj7")) == ["C4", "E4", "G4", "B4"]
    assert names(chord("G7")) == ["G4", "B4", "D5", "F5"]
    assert names(chord("Dm7")) == ["D4", "F4", "A4", "C5"]


def test_chord_with_explicit_octave() -> None:
    assert names(chord("C3maj7")) == ["C3", "E3", "G3", "B3"]


def test_chord_octave_argument() -> None:
    assert names(chord("C", octave=2)) == ["C2", "E2", "G2"]


def test_chord_extended_and_altered() -> None:
    assert names(chord("Cm7b5")) == ["C4", "D#4", "F#4", "A#4"]
    assert names(chord("Csus4")) == ["C4", "F4", "G4"]


def test_invalid_chord_rejected() -> None:
    with pytest.raises(ValueError, match="invalid chord"):
        chord("123")
    with pytest.raises(ValueError, match="unknown chord quality"):
        chord("Cbogus")
