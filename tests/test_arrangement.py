"""Tests for Pattern, Track timeline placement, and the Mixer."""

from __future__ import annotations

import numpy as np
import pytest

from codaw.arrangement.mixer import Mixer
from codaw.arrangement.pattern import Pattern
from codaw.arrangement.track import Track
from codaw.components.effects import Effect, Gain
from codaw.components.instrument import Synth
from codaw.core import oscillators
from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal
from codaw.music.pitch import Note
from codaw.music.timing import beats_to_seconds

SR = 44100


def plain_synth() -> Synth:
    # Flat envelope so amplitude is predictable for placement assertions.
    return Synth(oscillators.sine, ADSR(0.0, 0.0, 1.0, 0.0))


# -- Pattern ---------------------------------------------------------------


def test_pattern_from_string_places_sequentially() -> None:
    pat = Pattern.from_string("C4:1 E4:1 G4:2")
    assert len(pat) == 3
    starts = [e.start_beat for e in pat]
    assert starts == [0.0, 1.0, 2.0]
    assert pat.events[2].duration_beats == 2.0
    assert pat.length_beats == 4.0
    assert [e.pitch.name for e in pat] == ["C4", "E4", "G4"]


def test_pattern_from_string_velocity_and_rests() -> None:
    pat = Pattern.from_string("C4:1@0.5 _:1 E4:1")
    assert len(pat) == 2  # rest produces no event
    assert pat.events[0].velocity == 0.5
    assert pat.events[1].start_beat == 2.0  # pushed past the rest


def test_pattern_from_string_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="NOTE:DURATION"):
        Pattern.from_string("C4")
    with pytest.raises(ValueError, match="invalid token"):
        Pattern.from_string("C4:abc")


def test_pattern_sequence() -> None:
    pat = Pattern.sequence(["C4", "E4", "G4"], dur=0.5)
    assert len(pat) == 3
    assert [e.start_beat for e in pat] == [0.0, 0.5, 1.0]
    assert all(e.duration_beats == 0.5 for e in pat)


def test_pattern_add_and_iter() -> None:
    from codaw.components.events import NoteEvent

    pat = Pattern()
    pat.add(NoteEvent(Note(60), 0.0, 1.0))
    assert len(list(pat)) == 1
    assert "Pattern(" in repr(pat)


def test_empty_pattern_length() -> None:
    assert Pattern().length_beats == 0.0


# -- Track -----------------------------------------------------------------


def test_track_places_events_at_correct_sample_offsets() -> None:
    track = Track("t", plain_synth())
    track.add(Pattern.from_string("C4:1 C4:1"))  # two notes at beats 0 and 1
    out = track.render(bpm=120, sample_rate=SR)
    note_samples = int(beats_to_seconds(1.0, 120) * SR)  # 0.5 s = 22050
    # First note starts immediately (non-silent at 0); silence-then-sound at the
    # second note onset is non-silent too.
    assert np.max(np.abs(out.samples[:100])) > 0.1
    assert np.max(np.abs(out.samples[note_samples : note_samples + 100])) > 0.1
    assert out.length == 2 * note_samples


def test_track_offset_matches_beat_time() -> None:
    track = Track("t", plain_synth())
    # A rest then a note: the sound must begin at beat 2.
    track.add(Pattern.from_string("_:2 C4:1"))
    out = track.render(bpm=120, sample_rate=SR)
    onset = int(beats_to_seconds(2.0, 120) * SR)
    assert np.max(np.abs(out.samples[:onset])) < 1e-9  # silent before onset
    assert np.max(np.abs(out.samples[onset : onset + 100])) > 0.1


def test_track_velocity_scales_amplitude() -> None:
    loud = Track("l", plain_synth())
    loud.add(Pattern.from_string("C4:1@1.0"))
    soft = Track("s", plain_synth())
    soft.add(Pattern.from_string("C4:1@0.5"))
    lo = loud.render(120, SR)
    so = soft.render(120, SR)
    assert np.max(np.abs(so.samples)) == pytest.approx(0.5 * np.max(np.abs(lo.samples)), rel=1e-6)


def test_track_effect_chain_applies_in_order() -> None:
    class Record(Effect):
        def __init__(self, log: list[str], tag: str) -> None:
            self.log = log
            self.tag = tag

        def process(self, signal: Signal) -> Signal:
            self.log.append(self.tag)
            return signal

    log: list[str] = []
    track = Track("t", plain_synth(), effects=[Record(log, "a"), Record(log, "b")])
    track.add(Pattern.from_string("C4:1"))
    track.render(120, SR)
    assert log == ["a", "b"]


def test_track_gain_effect_halves_output() -> None:
    track = Track("t", plain_synth(), effects=[Gain(0.5)])
    track.add(Pattern.from_string("C4:1"))
    out = track.render(120, SR)
    assert np.max(np.abs(out.samples)) == pytest.approx(0.5, abs=0.02)


def test_empty_track_renders_silence() -> None:
    out = Track("t", plain_synth()).render(120, SR)
    assert out.length == 0


def test_track_validation_and_repr() -> None:
    with pytest.raises(ValueError, match="pan"):
        Track("t", plain_synth(), pan=2.0)
    with pytest.raises(ValueError, match="gain"):
        Track("t", plain_synth(), gain=-1.0)
    assert "Track(" in repr(Track("t", plain_synth()))


# -- Mixer -----------------------------------------------------------------


def test_mixer_sums_signals() -> None:
    a = Signal(np.full(10, 0.2), SR)
    b = Signal(np.full(10, 0.3), SR)
    out = Mixer().mix([a, b])
    np.testing.assert_allclose(out.samples, 0.5)


def test_mixer_applies_per_track_gain() -> None:
    a = Signal(np.ones(10), SR)
    out = Mixer().mix([a], gains=[0.25])
    np.testing.assert_allclose(out.samples, 0.25)


def test_mixer_pan_upmixes_to_stereo() -> None:
    a = Signal(np.full(10, 0.1), SR)
    out = Mixer().mix([a], pans=[-1.0])
    assert out.channels == 2
    assert np.allclose(out.samples[:, 1], 0.0, atol=1e-9)


def test_mixer_limiter_prevents_clipping() -> None:
    a = Signal(np.full(10, 0.8), SR)
    b = Signal(np.full(10, 0.8), SR)  # sum = 1.6 > ceiling
    out = Mixer(ceiling=0.99).mix([a, b])
    assert np.max(np.abs(out.samples)) == pytest.approx(0.99)


def test_mixer_quiet_mix_untouched() -> None:
    a = Signal(np.full(10, 0.1), SR)
    out = Mixer().mix([a])
    np.testing.assert_allclose(out.samples, 0.1)


def test_mixer_master_gain() -> None:
    a = Signal(np.full(10, 0.1), SR)
    out = Mixer(master_gain=2.0).mix([a])
    np.testing.assert_allclose(out.samples, 0.2)


def test_mixer_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Mixer().mix([])
    with pytest.raises(ValueError, match="gains"):
        Mixer().mix([Signal(np.ones(4), SR)], gains=[1.0, 2.0])
    with pytest.raises(ValueError, match="pans"):
        Mixer().mix([Signal(np.ones(4), SR)], pans=[0.0, 0.0])
    with pytest.raises(ValueError, match="master_gain"):
        Mixer(master_gain=-1.0)
    with pytest.raises(ValueError, match="ceiling"):
        Mixer(ceiling=0.0)
