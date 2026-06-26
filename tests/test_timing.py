"""Tests for beat/second timing conversions and Duration."""

from __future__ import annotations

import pytest

from codaw.music import timing
from codaw.music.timing import Duration, beats_to_seconds, seconds_to_beats


def test_beats_to_seconds_at_120bpm() -> None:
    # At 120 BPM, one beat (quarter note) is 0.5 s.
    assert beats_to_seconds(1.0, 120) == pytest.approx(0.5)
    assert beats_to_seconds(4.0, 120) == pytest.approx(2.0)


def test_beats_to_seconds_at_60bpm() -> None:
    assert beats_to_seconds(1.0, 60) == pytest.approx(1.0)


def test_seconds_to_beats_round_trips() -> None:
    for bpm in (60, 120, 90, 137):
        assert seconds_to_beats(beats_to_seconds(3.0, bpm), bpm) == pytest.approx(3.0)


def test_timing_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="bpm"):
        beats_to_seconds(1.0, 0)
    with pytest.raises(ValueError, match="beats"):
        beats_to_seconds(-1.0, 120)
    with pytest.raises(ValueError, match="bpm"):
        seconds_to_beats(1.0, -1)
    with pytest.raises(ValueError, match="seconds"):
        seconds_to_beats(-1.0, 120)


def test_note_value_constants() -> None:
    assert timing.WHOLE == 4.0
    assert timing.HALF == 2.0
    assert timing.QUARTER == 1.0
    assert timing.EIGHTH == 0.5
    assert timing.SIXTEENTH == 0.25
    assert timing.THIRTYSECOND == 0.125
    assert timing.DOTTED_QUARTER == 1.5
    assert timing.TRIPLET_EIGHTH == timing.EIGHTH * 2 / 3


def test_duration_seconds() -> None:
    assert Duration(2.0).seconds(120) == pytest.approx(1.0)


def test_duration_dotted_and_triplet() -> None:
    assert Duration(1.0).dotted().beats == pytest.approx(1.5)
    assert Duration(1.0).triplet().beats == pytest.approx(2 / 3)


def test_duration_arithmetic() -> None:
    assert (Duration(1.0) + Duration(0.5)).beats == pytest.approx(1.5)
    assert (Duration(1.0) * 3).beats == pytest.approx(3.0)
    assert (2 * Duration(1.5)).beats == pytest.approx(3.0)


def test_duration_equality_and_hash() -> None:
    assert Duration(1.0) == Duration(1.0)
    assert Duration(1.0) != Duration(2.0)
    assert len({Duration(1.0), Duration(1.0), Duration(2.0)}) == 2


def test_duration_rejects_negative() -> None:
    with pytest.raises(ValueError, match="beats"):
        Duration(-1.0)


def test_duration_repr() -> None:
    assert repr(Duration(1.5)) == "Duration(beats=1.5)"


def test_duration_arithmetic_with_wrong_type() -> None:
    assert Duration(1.0).__add__(5) is NotImplemented
    assert Duration(1.0).__mul__("x") is NotImplemented
    assert Duration(1.0).__eq__(5) is NotImplemented
