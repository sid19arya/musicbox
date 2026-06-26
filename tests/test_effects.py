"""Tests for the effects.

Each effect must change the signal in the asserted direction: Gain scales RMS,
Delay adds post-offset energy, Distortion adds harmonics, Filter attenuates its
stop-band, Pan produces stereo, Reverb adds a decaying tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from codaw.components.effects import (
    Delay,
    Distortion,
    Effect,
    Filter,
    Gain,
    Pan,
    Reverb,
)
from codaw.core import oscillators
from codaw.core.signal import Signal

SR = 44100


def rms(sig: Signal) -> float:
    return float(np.sqrt(np.mean(sig.samples**2)))


def harmonic_ratio(sig: Signal, fundamental: float) -> float:
    """Energy above the fundamental relative to total energy."""
    spectrum = np.abs(np.fft.rfft(sig.samples)) ** 2
    freqs = np.fft.rfftfreq(sig.length, 1.0 / sig.sample_rate)
    above = np.sum(spectrum[freqs > fundamental * 1.5])
    total = np.sum(spectrum) + 1e-12
    return float(above / total)


def test_effect_is_abstract() -> None:
    with pytest.raises(TypeError):
        Effect()  # type: ignore[abstract]


def test_gain_halves_rms() -> None:
    sig = oscillators.sine(440.0, 0.5, SR)
    out = Gain(0.5).process(sig)
    assert rms(out) == pytest.approx(0.5 * rms(sig), rel=1e-6)


def test_gain_rejects_negative() -> None:
    with pytest.raises(ValueError, match="factor"):
        Gain(-1.0)


def test_pan_produces_stereo() -> None:
    sig = oscillators.sine(440.0, 0.1, SR)
    out = Pan(-1.0).process(sig)
    assert out.channels == 2
    # Hard left: right channel silent.
    assert np.allclose(out.samples[:, 1], 0.0, atol=1e-9)


def test_pan_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="position"):
        Pan(2.0)


def test_delay_adds_energy_after_offset() -> None:
    # A short click followed by silence; the echo must appear one delay later.
    samples = np.zeros(SR)
    samples[:100] = 1.0
    sig = Signal(samples, SR)
    out = Delay(time=0.25, feedback=0.0, mix=1.0).process(sig)
    offset = int(0.25 * SR)
    # Region around the echo offset is non-silent in the output.
    echo_region = out.samples[offset : offset + 100]
    assert np.max(np.abs(echo_region)) > 0.5


def test_delay_feedback_creates_multiple_echoes() -> None:
    samples = np.zeros(SR // 2)
    samples[:50] = 1.0
    sig = Signal(samples, SR)
    out = Delay(time=0.1, feedback=0.5, mix=1.0).process(sig)
    n = int(0.1 * SR)
    first = np.max(np.abs(out.samples[n : n + 50]))
    second = np.max(np.abs(out.samples[2 * n : 2 * n + 50]))
    assert first > 0.0
    assert second > 0.0
    assert second < first  # decaying


def test_delay_extends_length_for_tail() -> None:
    sig = oscillators.sine(440.0, 0.2, SR)
    out = Delay(time=0.1, feedback=0.3, mix=0.5).process(sig)
    assert out.length > sig.length


def test_delay_validation() -> None:
    with pytest.raises(ValueError, match="time"):
        Delay(time=0.0)
    with pytest.raises(ValueError, match="feedback"):
        Delay(time=0.1, feedback=1.0)
    with pytest.raises(ValueError, match="mix"):
        Delay(time=0.1, mix=2.0)


def test_distortion_increases_harmonics() -> None:
    sig = oscillators.sine(220.0, 0.5, SR)
    out = Distortion(drive=10.0).process(sig)
    assert harmonic_ratio(out, 220.0) > harmonic_ratio(sig, 220.0) + 0.05


def test_distortion_preserves_length() -> None:
    sig = oscillators.sine(220.0, 0.3, SR)
    out = Distortion(drive=5.0).process(sig)
    assert out.length == sig.length


def test_distortion_rejects_non_positive_drive() -> None:
    with pytest.raises(ValueError, match="drive"):
        Distortion(drive=0.0)


def test_distortion_handles_silence() -> None:
    out = Distortion(drive=5.0).process(Signal(np.zeros(100), SR))
    assert np.all(out.samples == 0.0)


def test_filter_effect_lowpass_attenuates_highs() -> None:
    sig = oscillators.sine(200.0, 0.5, SR) + oscillators.sine(8000.0, 0.5, SR)
    out = Filter("lowpass", 1000.0).process(sig)
    spectrum = np.abs(np.fft.rfft(out.samples))
    freqs = np.fft.rfftfreq(out.length, 1.0 / SR)
    high = np.max(spectrum[(freqs > 7000) & (freqs < 9000)])
    low = np.max(spectrum[(freqs > 100) & (freqs < 300)])
    assert high < 0.1 * low


def test_filter_effect_highpass_attenuates_lows() -> None:
    sig = oscillators.sine(200.0, 0.5, SR) + oscillators.sine(8000.0, 0.5, SR)
    out = Filter("highpass", 2000.0).process(sig)
    spectrum = np.abs(np.fft.rfft(out.samples))
    freqs = np.fft.rfftfreq(out.length, 1.0 / SR)
    low = np.max(spectrum[(freqs > 100) & (freqs < 300)])
    high = np.max(spectrum[(freqs > 7000) & (freqs < 9000)])
    assert low < 0.1 * high


def test_filter_effect_bandpass() -> None:
    sig = oscillators.sine(1000.0, 0.3, SR)
    out = Filter("bandpass", (500.0, 2000.0)).process(sig)
    assert out.length == sig.length


def test_filter_validation() -> None:
    with pytest.raises(ValueError, match="kind"):
        Filter("notch", 1000.0)
    with pytest.raises(ValueError, match="tuple"):
        Filter("bandpass", 1000.0)
    with pytest.raises(ValueError, match="single"):
        Filter("lowpass", (500.0, 2000.0))


def test_reverb_adds_tail() -> None:
    samples = np.zeros(SR // 4)
    samples[:100] = 1.0
    sig = Signal(samples, SR)
    out = Reverb(room_size=0.7, mix=0.5).process(sig)
    assert out.length > sig.length
    # There is reverberant energy after the original click ends.
    tail = out.samples[SR // 4 :]
    assert np.max(np.abs(tail)) > 0.0


def test_reverb_dry_mix_changes_signal() -> None:
    sig = oscillators.sine(440.0, 0.2, SR)
    out = Reverb(room_size=0.5, mix=0.4).process(sig)
    # Output differs from the (padded) dry signal.
    dry = sig.pad_to(out.length).samples
    assert not np.allclose(out.samples, dry)


def test_reverb_validation() -> None:
    with pytest.raises(ValueError, match="room_size"):
        Reverb(room_size=1.5)
    with pytest.raises(ValueError, match="mix"):
        Reverb(mix=-0.1)


def test_effects_chain_in_order() -> None:
    sig = oscillators.sine(440.0, 0.2, SR)
    chain: list[Effect] = [Gain(0.5), Distortion(drive=3.0)]
    out = sig
    for effect in chain:
        out = effect.process(out)
    assert out.length == sig.length
