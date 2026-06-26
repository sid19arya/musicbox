"""Tests for the Note pitch type."""

from __future__ import annotations

import pytest

from codaw.music.pitch import Note

# (name, midi, frequency) reference table.
KNOWN = [
    ("C-1", 0, 8.1757989156),
    ("C4", 60, 261.6255653006),
    ("A4", 69, 440.0),
    ("A#4", 70, 466.1637615181),
    ("C5", 72, 523.2511306012),
    ("A0", 21, 27.5),
    ("C8", 108, 4186.009044809),
]


@pytest.mark.parametrize(("name", "midi", "freq"), KNOWN)
def test_name_to_midi(name: str, midi: int, freq: float) -> None:
    assert Note.from_name(name).midi == midi


@pytest.mark.parametrize(("name", "midi", "freq"), KNOWN)
def test_midi_to_frequency(name: str, midi: int, freq: float) -> None:
    assert Note(midi).frequency == pytest.approx(freq)


def test_a4_is_exactly_440() -> None:
    assert Note.from_name("A4").frequency == 440.0


def test_name_midi_freq_round_trip() -> None:
    for midi in range(0, 128):
        note = Note(midi)
        assert Note.from_name(note.name).midi == midi


def test_flats_and_sharps_are_enharmonic() -> None:
    assert Note.from_name("A#3").midi == Note.from_name("Bb3").midi
    assert Note.from_name("Db4").midi == Note.from_name("C#4").midi


def test_double_accidentals() -> None:
    assert Note.from_name("Cbb4").midi == Note.from_name("Bb3").midi
    assert Note.from_name("C##4").midi == Note.from_name("D4").midi


def test_canonical_name_uses_sharps() -> None:
    assert Note(61).name == "C#4"
    assert Note(60).name == "C4"


def test_transpose() -> None:
    assert Note.from_name("C4").transpose(12) == Note.from_name("C5")
    assert Note.from_name("C4").transpose(-1) == Note.from_name("B3")


def test_coerce_accepts_note_str_int() -> None:
    assert Note.coerce(Note(60)).midi == 60
    assert Note.coerce("C4").midi == 60
    assert Note.coerce(60).midi == 60


def test_coerce_rejects_other() -> None:
    with pytest.raises(TypeError):
        Note.coerce(60.5)  # type: ignore[arg-type]


def test_invalid_name_rejected() -> None:
    with pytest.raises(ValueError, match="invalid note"):
        Note.from_name("H4")
    with pytest.raises(ValueError, match="invalid note"):
        Note.from_name("C")


def test_non_int_midi_rejected() -> None:
    with pytest.raises(TypeError):
        Note(60.0)  # type: ignore[arg-type]


def test_str() -> None:
    assert str(Note(69)) == "A4"
