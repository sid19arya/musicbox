"""Tests for the Instrument ABC, Synth, and NoteEvent."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from codaw.components.events import NoteEvent
from codaw.components.instrument import Instrument, Synth
from codaw.core import oscillators
from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal
from codaw.music.pitch import Note

SR = 44100


def dominant_freq(sig: Signal) -> float:
    spectrum = np.abs(np.fft.rfft(sig.samples))
    freqs = np.fft.rfftfreq(sig.length, 1.0 / sig.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def test_instrument_is_abstract() -> None:
    with pytest.raises(TypeError):
        Instrument()  # type: ignore[abstract]


def test_synth_render_length_and_pitch() -> None:
    synth = Synth(oscillators.sine, ADSR(0.01, 0.1, 0.7, 0.1))
    sig = synth.render(Note.from_name("A4"), 1.0, SR)
    assert sig.length == SR
    assert sig.channels == 1
    assert abs(dominant_freq(sig) - 440.0) < 2.0


def test_synth_render_is_non_silent() -> None:
    synth = Synth(oscillators.saw)
    sig = synth.render(Note.from_name("C4"), 0.5, SR)
    assert np.max(np.abs(sig.samples)) > 0.1


def test_synth_applies_envelope() -> None:
    synth = Synth(oscillators.sine, ADSR(0.05, 0.1, 0.7, 0.1))
    sig = synth.render(Note.from_name("A4"), 1.0, SR)
    # Envelope forces the very first and last samples toward zero.
    assert abs(sig.samples[0]) < 1e-6
    assert abs(sig.samples[-1]) < 1e-3


def test_synth_gain_scales_output() -> None:
    note = Note.from_name("A4")
    loud = Synth(oscillators.sine, ADSR(0.0, 0.0, 1.0, 0.0), gain=1.0).render(note, 0.5, SR)
    quiet = Synth(oscillators.sine, ADSR(0.0, 0.0, 1.0, 0.0), gain=0.5).render(note, 0.5, SR)
    assert np.max(np.abs(quiet.samples)) == pytest.approx(
        0.5 * np.max(np.abs(loud.samples)), rel=1e-6
    )


def test_synth_default_envelope() -> None:
    synth = Synth(oscillators.sine)
    assert isinstance(synth.envelope, ADSR)
    sig = synth.render(Note.from_name("A4"), 0.3, SR)
    assert sig.length == int(0.3 * SR)


def test_synth_rejects_bad_gain_and_duration() -> None:
    with pytest.raises(ValueError, match="gain"):
        Synth(oscillators.sine, gain=-1.0)
    with pytest.raises(ValueError, match="duration"):
        Synth(oscillators.sine).render(Note.from_name("A4"), 0.0, SR)


def test_custom_instrument_subclass() -> None:
    class Silence(Instrument):
        def render(self, note: Note, duration_s: float, sample_rate: int = 44100) -> Signal:
            return Signal(np.zeros(int(duration_s * sample_rate)), sample_rate)

    sig = Silence().render(Note(60), 0.1, SR)
    assert sig.length == int(0.1 * SR)
    assert np.all(sig.samples == 0.0)


def test_note_event_fields_and_end() -> None:
    ev = NoteEvent(Note.from_name("C4"), start_beat=1.0, duration_beats=2.0, velocity=0.8)
    assert ev.pitch.name == "C4"
    assert ev.end_beat == 3.0
    assert ev.velocity == 0.8


def test_note_event_coerces_pitch() -> None:
    ev = NoteEvent("E4", 0.0, 1.0)  # type: ignore[arg-type]
    assert isinstance(ev.pitch, Note)
    assert ev.pitch.name == "E4"
    ev_midi = NoteEvent(60, 0.0, 1.0)  # type: ignore[arg-type]
    assert ev_midi.pitch.name == "C4"


def test_note_event_default_velocity() -> None:
    ev = NoteEvent(Note(60), 0.0, 1.0)
    assert ev.velocity == 1.0


def test_note_event_validation() -> None:
    with pytest.raises(ValueError, match="start_beat"):
        NoteEvent(Note(60), -1.0, 1.0)
    with pytest.raises(ValueError, match="duration_beats"):
        NoteEvent(Note(60), 0.0, 0.0)
    with pytest.raises(ValueError, match="velocity"):
        NoteEvent(Note(60), 0.0, 1.0, velocity=1.5)


def test_note_event_is_frozen() -> None:
    ev = NoteEvent(Note(60), 0.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        ev.velocity = 0.5  # type: ignore[misc]
