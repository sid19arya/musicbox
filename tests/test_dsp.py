"""Tests for the dsp helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from codaw.core import dsp
from codaw.core.signal import Signal

SR = 44100


def test_gain_scales_amplitude() -> None:
    sig = Signal(np.ones(10), SR)
    assert np.allclose(dsp.gain(sig, 0.25).samples, 0.25)


def test_mix_sums_and_aligns() -> None:
    a = Signal(np.ones(6), SR)
    b = Signal(np.ones(3), SR)
    c = Signal(np.full(4, 2.0), SR)
    out = dsp.mix(a, b, c)
    assert out.length == 6
    # First 3 samples: 1+1+2 = 4.
    np.testing.assert_allclose(out.samples[:3], 4.0)


def test_mix_single_signal() -> None:
    a = Signal(np.ones(5), SR)
    np.testing.assert_allclose(dsp.mix(a).samples, 1.0)


def test_mix_requires_signal() -> None:
    with pytest.raises(ValueError, match="at least one"):
        dsp.mix()


def test_normalize_hits_target_peak() -> None:
    sig = Signal(np.array([0.1, -0.3, 0.2]), SR)
    out = dsp.normalize(sig, peak=0.9)
    assert np.max(np.abs(out.samples)) == pytest.approx(0.9)


def test_normalize_silent_signal_unchanged() -> None:
    sig = Signal(np.zeros(10), SR)
    out = dsp.normalize(sig, peak=0.9)
    np.testing.assert_allclose(out.samples, 0.0)


def test_normalize_empty_signal_unchanged() -> None:
    sig = Signal(np.zeros(0), SR)
    out = dsp.normalize(sig)
    assert out.length == 0


def test_normalize_rejects_negative_peak() -> None:
    with pytest.raises(ValueError, match="peak"):
        dsp.normalize(Signal(np.ones(4), SR), peak=-1.0)


def test_clip_limits_range() -> None:
    sig = Signal(np.array([2.0, -2.0, 0.5]), SR)
    out = dsp.clip(sig, threshold=1.0)
    np.testing.assert_allclose(out.samples, [1.0, -1.0, 0.5])


def test_clip_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        dsp.clip(Signal(np.ones(4), SR), threshold=-0.5)


def test_pan_center_is_equal_power() -> None:
    sig = Signal(np.ones(8), SR)
    out = dsp.pan(sig, 0.0)
    assert out.channels == 2
    np.testing.assert_allclose(out.samples[:, 0], out.samples[:, 1])
    assert out.samples[0, 0] == pytest.approx(np.sqrt(0.5), abs=1e-9)


def test_pan_hard_left_and_right() -> None:
    sig = Signal(np.ones(8), SR)
    left = dsp.pan(sig, -1.0)
    assert np.allclose(left.samples[:, 0], 1.0)
    assert np.allclose(left.samples[:, 1], 0.0, atol=1e-9)
    right = dsp.pan(sig, 1.0)
    assert np.allclose(right.samples[:, 0], 0.0, atol=1e-9)
    assert np.allclose(right.samples[:, 1], 1.0)


def test_pan_downmixes_stereo_input() -> None:
    stereo = Signal(np.ones((8, 2)), SR)
    out = dsp.pan(stereo, 0.0)
    assert out.channels == 2


def test_pan_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="position"):
        dsp.pan(Signal(np.ones(4), SR), 2.0)


def test_fade_in_starts_silent_and_recovers() -> None:
    sig = Signal(np.ones(SR), SR)
    out = dsp.fade_in(sig, 0.1)
    assert out.samples[0] == pytest.approx(0.0)
    assert out.samples[-1] == pytest.approx(1.0)
    n = int(0.1 * SR)
    assert np.all(np.diff(out.samples[:n]) >= 0)


def test_fade_out_ends_silent() -> None:
    sig = Signal(np.ones(SR), SR)
    out = dsp.fade_out(sig, 0.1)
    assert out.samples[0] == pytest.approx(1.0)
    assert out.samples[-1] == pytest.approx(0.0)


def test_fade_preserves_stereo() -> None:
    sig = Signal(np.ones((SR, 2)), SR)
    out = dsp.fade_in(sig, 0.05)
    assert out.channels == 2
    assert out.samples[0, 0] == pytest.approx(0.0)


def test_fade_zero_duration_is_noop() -> None:
    sig = Signal(np.ones(100), SR)
    np.testing.assert_allclose(dsp.fade_in(sig, 0.0).samples, 1.0)


def test_fade_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        dsp.fade_out(Signal(np.ones(4), SR), -0.1)
    with pytest.raises(ValueError, match="duration"):
        dsp.fade_in(Signal(np.ones(4), SR), -0.1)
