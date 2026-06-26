"""MVP smoke tests: the thin end-to-end slice must hold together.

Covers the :class:`Signal` primitive, the :func:`sine` oscillator and the
``Song`` construct → render → save WAV path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from codaw import Signal, Song, oscillators


def test_signal_basic_properties() -> None:
    sig = Signal(np.zeros(44100), sample_rate=44100)
    assert sig.length == 44100
    assert sig.channels == 1
    assert sig.duration == 1.0
    assert len(sig) == 44100
    assert "Signal(" in repr(sig)


def test_signal_stereo_channels() -> None:
    sig = Signal(np.zeros((100, 2)), sample_rate=44100)
    assert sig.channels == 2
    assert sig.length == 100


def test_signal_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError, match="1-D"):
        Signal(np.zeros((2, 2, 2)), sample_rate=44100)


def test_signal_rejects_bad_channel_count() -> None:
    with pytest.raises(ValueError, match="channels"):
        Signal(np.zeros((100, 3)), sample_rate=44100)


def test_signal_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        Signal(np.zeros(10), sample_rate=0)


def test_sine_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        oscillators.sine(440.0, -1.0, 44100)


def test_song_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="bpm"):
        Song(bpm=0)
    with pytest.raises(ValueError, match="sample_rate"):
        Song(bpm=120, sample_rate=-1)


def test_signal_copies_input_array() -> None:
    arr = np.ones(10)
    sig = Signal(arr, sample_rate=10)
    arr[0] = 999.0
    assert sig.samples[0] == 1.0  # not mutated by caller


def test_sine_length_and_peak() -> None:
    sig = oscillators.sine(440.0, 1.0, 44100)
    assert sig.length == 44100
    assert np.isclose(np.max(np.abs(sig.samples)), 1.0, atol=1e-3)


def test_sine_dominant_frequency() -> None:
    sr = 44100
    sig = oscillators.sine(440.0, 1.0, sr)
    spectrum = np.abs(np.fft.rfft(sig.samples))
    freqs = np.fft.rfftfreq(sig.length, 1.0 / sr)
    dominant = freqs[int(np.argmax(spectrum))]
    assert abs(dominant - 440.0) < 1.0


def test_song_render_returns_signal() -> None:
    song = Song(bpm=120, sample_rate=44100)
    sig = song.render()
    assert isinstance(sig, Signal)
    assert sig.length == 44100
    assert np.any(sig.samples != 0.0)  # non-silent


def test_song_save_writes_valid_wav(tmp_path: Path) -> None:
    song = Song(bpm=120, sample_rate=44100)
    out = song.save(tmp_path / "out.wav")
    assert out.exists()
    data, sr = sf.read(str(out))
    assert sr == 44100
    assert len(data) == 44100
    assert np.any(data != 0.0)
