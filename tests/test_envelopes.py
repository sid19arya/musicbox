"""Tests for the ADSR envelope.

Assert the canonical envelope shape: starts near 0, rises through attack,
holds at the sustain level, and decays toward 0 in the release tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal

SR = 44100


def dc(length: int, sample_rate: int = SR) -> Signal:
    """A constant unit (DC) signal so the envelope curve is visible directly."""
    return Signal(np.ones(length), sample_rate)


def test_envelope_starts_at_zero_and_rises() -> None:
    env = ADSR(0.1, 0.1, 0.7, 0.1)
    curve = env.envelope(SR, SR)
    assert curve[0] == pytest.approx(0.0, abs=1e-9)
    # During attack the curve is strictly increasing.
    attack_n = int(0.1 * SR)
    assert np.all(np.diff(curve[:attack_n]) > 0)
    # Attack reaches ~1 at its peak.
    assert curve[:attack_n].max() == pytest.approx(1.0, abs=0.01)


def test_envelope_reaches_sustain_level() -> None:
    env = ADSR(0.05, 0.05, 0.6, 0.05)
    curve = env.envelope(SR, SR)
    # Mid-signal (well past decay, before release) should sit at sustain.
    mid = curve[SR // 2]
    assert mid == pytest.approx(0.6, abs=1e-6)


def test_envelope_decays_to_zero_at_end() -> None:
    env = ADSR(0.01, 0.05, 0.7, 0.2)
    curve = env.envelope(SR, SR)
    assert curve[-1] == pytest.approx(0.0, abs=1e-6)
    # Release tail is strictly decreasing.
    release_n = int(0.2 * SR)
    tail = curve[-release_n:]
    assert np.all(np.diff(tail) < 0)


def test_apply_multiplies_signal() -> None:
    env = ADSR(0.01, 0.1, 0.7, 0.2)
    sig = dc(SR)
    out = env.apply(sig)
    assert isinstance(out, Signal)
    assert out.length == sig.length
    assert out.samples[0] == pytest.approx(0.0, abs=1e-9)
    assert out.samples[-1] == pytest.approx(0.0, abs=1e-6)
    # Envelope is a gain in [0, 1]: never amplifies a unit signal.
    assert np.max(np.abs(out.samples)) <= 1.0 + 1e-9


def test_apply_preserves_stereo() -> None:
    env = ADSR(0.01, 0.05, 0.8, 0.05)
    stereo = Signal(np.ones((SR, 2)), SR)
    out = env.apply(stereo)
    assert out.channels == 2
    assert out.length == SR
    # Both channels get the same curve.
    np.testing.assert_allclose(out.samples[:, 0], out.samples[:, 1])


def test_zero_attack_starts_at_full() -> None:
    env = ADSR(0.0, 0.0, 1.0, 0.0)
    curve = env.envelope(100, SR)
    # No attack/decay/release, sustain=1 → flat unity gain.
    np.testing.assert_allclose(curve, np.ones(100))


def test_release_longer_than_signal_is_clamped() -> None:
    env = ADSR(0.0, 0.0, 1.0, 10.0)
    curve = env.envelope(100, SR)
    # Whole signal becomes the release ramp down to 0.
    assert curve[0] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(curve) <= 0)


def test_empty_signal_returns_empty_curve() -> None:
    env = ADSR(0.01, 0.01, 0.5, 0.01)
    assert env.envelope(0, SR).shape == (0,)


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ADSR(-0.1, 0.1, 0.5, 0.1)
    with pytest.raises(ValueError, match="sustain"):
        ADSR(0.1, 0.1, 1.5, 0.1)


def test_envelope_rejects_bad_sample_rate() -> None:
    env = ADSR(0.01, 0.01, 0.5, 0.01)
    with pytest.raises(ValueError, match="sample_rate"):
        env.envelope(100, 0)
