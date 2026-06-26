"""Tests for the SciPy-backed filters.

Assert spectral attenuation: a filter must reduce energy in its stop-band
relative to the unfiltered signal while largely preserving the pass-band.
"""

from __future__ import annotations

import numpy as np
import pytest

from codaw.core import filters, oscillators
from codaw.core.signal import Signal

SR = 44100


def band_energy(sig: Signal, lo: float, hi: float) -> float:
    """Total spectral energy of a mono signal within ``[lo, hi]`` Hz."""
    spectrum = np.abs(np.fft.rfft(sig.samples))
    freqs = np.fft.rfftfreq(sig.length, 1.0 / sig.sample_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(spectrum[mask] ** 2))


def two_tone(low_hz: float, high_hz: float) -> Signal:
    """A signal mixing a low and a high tone of equal amplitude."""
    return oscillators.sine(low_hz, 1.0, SR) + oscillators.sine(high_hz, 1.0, SR)


def test_lowpass_attenuates_highs() -> None:
    sig = two_tone(200.0, 8000.0)
    out = filters.lowpass(sig, cutoff=1000.0)
    # High tone energy is strongly reduced; low tone largely preserved.
    assert band_energy(out, 6000.0, 10000.0) < 0.01 * band_energy(sig, 6000.0, 10000.0)
    assert band_energy(out, 100.0, 400.0) > 0.5 * band_energy(sig, 100.0, 400.0)


def test_highpass_attenuates_lows() -> None:
    sig = two_tone(200.0, 8000.0)
    out = filters.highpass(sig, cutoff=2000.0)
    assert band_energy(out, 100.0, 400.0) < 0.01 * band_energy(sig, 100.0, 400.0)
    assert band_energy(out, 6000.0, 10000.0) > 0.5 * band_energy(sig, 6000.0, 10000.0)


def test_bandpass_keeps_mid_drops_edges() -> None:
    low = oscillators.sine(100.0, 1.0, SR)
    mid = oscillators.sine(1000.0, 1.0, SR)
    high = oscillators.sine(10000.0, 1.0, SR)
    sig = low + mid + high
    out = filters.bandpass(sig, low=500.0, high=2000.0)
    assert band_energy(out, 50.0, 200.0) < 0.05 * band_energy(sig, 50.0, 200.0)
    assert band_energy(out, 8000.0, 12000.0) < 0.05 * band_energy(sig, 8000.0, 12000.0)
    assert band_energy(out, 800.0, 1200.0) > 0.5 * band_energy(sig, 800.0, 1200.0)


def test_filters_preserve_length_and_rate() -> None:
    sig = oscillators.sine(440.0, 0.5, SR)
    for out in (
        filters.lowpass(sig, 1000.0),
        filters.highpass(sig, 1000.0),
        filters.bandpass(sig, 500.0, 2000.0),
    ):
        assert out.length == sig.length
        assert out.sample_rate == SR
        assert out.channels == 1


def test_filters_handle_stereo() -> None:
    sig = two_tone(200.0, 8000.0).to_stereo()
    out = filters.lowpass(sig, 1000.0)
    assert out.channels == 2
    assert out.length == sig.length


def test_lowpass_rejects_bad_cutoff() -> None:
    sig = oscillators.sine(440.0, 0.1, SR)
    with pytest.raises(ValueError, match="cutoff"):
        filters.lowpass(sig, cutoff=0.0)
    with pytest.raises(ValueError, match="cutoff"):
        filters.lowpass(sig, cutoff=SR)  # above Nyquist


def test_highpass_rejects_bad_cutoff() -> None:
    sig = oscillators.sine(440.0, 0.1, SR)
    with pytest.raises(ValueError, match="cutoff"):
        filters.highpass(sig, cutoff=30000.0)


def test_bandpass_rejects_bad_band() -> None:
    sig = oscillators.sine(440.0, 0.1, SR)
    with pytest.raises(ValueError, match="low"):
        filters.bandpass(sig, low=2000.0, high=500.0)
