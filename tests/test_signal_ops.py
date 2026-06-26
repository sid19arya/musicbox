"""Tests for Signal arithmetic, shaping and channel conversion."""

from __future__ import annotations

import numpy as np
import pytest

from codaw.core.signal import Signal

SR = 44100


def test_scalar_gain_via_mul() -> None:
    sig = Signal(np.ones(10), SR)
    out = sig * 0.5
    np.testing.assert_allclose(out.samples, 0.5)
    # __rmul__ too.
    np.testing.assert_allclose((2.0 * sig).samples, 2.0)


def test_mul_rejects_non_scalar() -> None:
    sig = Signal(np.ones(10), SR)
    with pytest.raises(TypeError):
        _ = sig * sig  # type: ignore[operator]


def test_add_same_length_mono() -> None:
    a = Signal(np.ones(10), SR)
    b = Signal(np.full(10, 2.0), SR)
    np.testing.assert_allclose((a + b).samples, 3.0)


def test_add_length_aligns_with_zero_pad() -> None:
    a = Signal(np.ones(10), SR)
    b = Signal(np.ones(4), SR)
    out = a + b
    assert out.length == 10
    np.testing.assert_allclose(out.samples[:4], 2.0)
    np.testing.assert_allclose(out.samples[4:], 1.0)


def test_add_upmixes_mono_with_stereo() -> None:
    mono = Signal(np.ones(8), SR)
    stereo = Signal(np.full((8, 2), 2.0), SR)
    out = mono + stereo
    assert out.channels == 2
    np.testing.assert_allclose(out.samples, 3.0)


def test_add_rejects_sample_rate_mismatch() -> None:
    a = Signal(np.ones(4), 44100)
    b = Signal(np.ones(4), 22050)
    with pytest.raises(ValueError, match="sample rate"):
        _ = a + b


def test_add_rejects_non_signal() -> None:
    a = Signal(np.ones(4), SR)
    with pytest.raises(TypeError):
        _ = a + 5  # type: ignore[operator]


def test_to_stereo_duplicates_mono() -> None:
    mono = Signal(np.array([1.0, 2.0, 3.0]), SR)
    stereo = mono.to_stereo()
    assert stereo.channels == 2
    np.testing.assert_allclose(stereo.samples[:, 0], stereo.samples[:, 1])
    np.testing.assert_allclose(stereo.samples[:, 0], [1.0, 2.0, 3.0])


def test_to_stereo_on_stereo_is_copy() -> None:
    stereo = Signal(np.ones((4, 2)), SR)
    out = stereo.to_stereo()
    assert out.channels == 2
    assert out.samples is not stereo.samples


def test_to_mono_averages_channels() -> None:
    stereo = Signal(np.array([[0.0, 2.0], [4.0, 6.0]]), SR)
    mono = stereo.to_mono()
    assert mono.channels == 1
    np.testing.assert_allclose(mono.samples, [1.0, 5.0])


def test_to_mono_on_mono_is_copy() -> None:
    mono = Signal(np.array([1.0, 2.0]), SR)
    out = mono.to_mono()
    np.testing.assert_allclose(out.samples, [1.0, 2.0])


def test_pad_to_extends_mono() -> None:
    sig = Signal(np.ones(4), SR)
    out = sig.pad_to(10)
    assert out.length == 10
    np.testing.assert_allclose(out.samples[4:], 0.0)


def test_pad_to_extends_stereo() -> None:
    sig = Signal(np.ones((4, 2)), SR)
    out = sig.pad_to(10)
    assert out.length == 10
    assert out.channels == 2
    np.testing.assert_allclose(out.samples[4:], 0.0)


def test_pad_to_shorter_is_noop_copy() -> None:
    sig = Signal(np.ones(10), SR)
    out = sig.pad_to(4)
    assert out.length == 10


def test_pad_to_rejects_negative() -> None:
    sig = Signal(np.ones(4), SR)
    with pytest.raises(ValueError, match="non-negative"):
        sig.pad_to(-1)


def test_slice_extracts_time_window() -> None:
    sig = Signal(np.arange(SR, dtype=float), SR)
    out = sig.slice(0.25, 0.75)
    assert out.length == SR // 2
    assert out.samples[0] == pytest.approx(SR * 0.25)


def test_slice_clamps_to_extent() -> None:
    sig = Signal(np.ones(100), SR)
    out = sig.slice(0.0, 10.0)  # end far beyond the signal
    assert out.length == 100


def test_slice_rejects_bad_bounds() -> None:
    sig = Signal(np.ones(100), SR)
    with pytest.raises(ValueError, match="non-negative"):
        sig.slice(-0.1, 0.5)
    with pytest.raises(ValueError, match=">="):
        sig.slice(0.5, 0.1)
