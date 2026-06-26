"""Tests for the oscillator waveforms.

Each waveform must have the correct length, the expected dominant frequency,
and a peak near 1.0. Noise must be reproducible for a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from codaw.core import oscillators
from codaw.core.signal import Signal

SR = 44100
WAVE_GENERATORS = [oscillators.sine, oscillators.saw, oscillators.square, oscillators.triangle]


def dominant_freq(sig: Signal) -> float:
    """Return the dominant FFT frequency of a mono signal."""
    spectrum = np.abs(np.fft.rfft(sig.samples))
    freqs = np.fft.rfftfreq(sig.length, 1.0 / sig.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


@pytest.mark.parametrize("generator", WAVE_GENERATORS)
def test_waveform_length(generator) -> None:
    sig = generator(440.0, 1.0, SR)
    assert sig.length == SR
    assert sig.channels == 1


@pytest.mark.parametrize("generator", WAVE_GENERATORS)
def test_waveform_dominant_frequency(generator) -> None:
    sig = generator(440.0, 1.0, SR)
    assert abs(dominant_freq(sig) - 440.0) < 1.0


@pytest.mark.parametrize("generator", WAVE_GENERATORS)
def test_waveform_peak_near_one(generator) -> None:
    sig = generator(110.0, 0.5, SR)
    assert np.max(np.abs(sig.samples)) <= 1.0 + 1e-9
    assert np.max(np.abs(sig.samples)) > 0.9


def test_sine_values() -> None:
    # A 1 Hz sine at 4 samples/s: [0, 1, 0, -1].
    sig = oscillators.sine(1.0, 1.0, 4)
    np.testing.assert_allclose(sig.samples, [0.0, 1.0, 0.0, -1.0], atol=1e-9)


def test_saw_ramps_from_negative_to_positive() -> None:
    sig = oscillators.saw(1.0, 1.0, 8)
    # One cycle over 8 samples: starts at -1, rises monotonically within the cycle.
    assert sig.samples[0] == pytest.approx(-1.0)
    assert np.all(np.diff(sig.samples) > 0)


def test_square_is_bipolar_and_respects_duty() -> None:
    sig = oscillators.square(1.0, 1.0, 100)
    assert set(np.unique(sig.samples)).issubset({-1.0, 1.0})
    # Default duty 0.5 → equal split.
    assert np.sum(sig.samples > 0) == pytest.approx(50, abs=1)

    quarter = oscillators.square(1.0, 1.0, 100, duty=0.25)
    assert np.sum(quarter.samples > 0) == pytest.approx(25, abs=1)


def test_square_rejects_bad_duty() -> None:
    with pytest.raises(ValueError, match="duty"):
        oscillators.square(440.0, 1.0, SR, duty=0.0)


def test_triangle_peaks_at_plus_one() -> None:
    sig = oscillators.triangle(1.0, 1.0, 100)
    assert sig.samples[0] == pytest.approx(-1.0, abs=1e-9)
    assert np.max(sig.samples) == pytest.approx(1.0, abs=0.05)
    assert np.min(sig.samples) == pytest.approx(-1.0, abs=1e-9)


def test_noise_in_range_and_correct_length() -> None:
    sig = oscillators.noise(0.5, SR, seed=42)
    assert sig.length == SR // 2
    assert np.all(sig.samples >= -1.0)
    assert np.all(sig.samples <= 1.0)


def test_noise_is_reproducible_for_fixed_seed() -> None:
    a = oscillators.noise(0.25, SR, seed=7)
    b = oscillators.noise(0.25, SR, seed=7)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_noise_differs_across_seeds() -> None:
    a = oscillators.noise(0.25, SR, seed=1)
    b = oscillators.noise(0.25, SR, seed=2)
    assert not np.array_equal(a.samples, b.samples)


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValueError, match="duration"):
        oscillators.saw(440.0, -0.1, SR)


def test_zero_duration_is_empty() -> None:
    sig = oscillators.sine(440.0, 0.0, SR)
    assert sig.length == 0


def test_non_positive_sample_rate_rejected() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        oscillators.sine(440.0, 1.0, 0)
