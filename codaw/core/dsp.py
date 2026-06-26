"""DSP helpers operating on :class:`Signal` objects.

Pure functions: each returns a new :class:`Signal` and never mutates its input.
These wrap the common gain-staging and shaping operations used throughout the
arrangement and effects layers.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import numpy.typing as npt

from codaw.core.signal import Signal

__all__ = [
    "gain",
    "mix",
    "normalize",
    "clip",
    "pan",
    "fade_in",
    "fade_out",
]


def gain(signal: Signal, factor: float) -> Signal:
    """Scale a signal's amplitude by a linear ``factor``."""
    return signal * factor


def mix(*signals: Signal) -> Signal:
    """Sum any number of signals, length- and channel-aligning them.

    Parameters
    ----------
    *signals:
        One or more signals sharing a sample rate. Shorter signals are
        zero-padded; if any is stereo the result is stereo.

    Returns
    -------
    Signal
        The length-aligned sum.

    Raises
    ------
    ValueError
        If no signals are given.
    """
    if not signals:
        raise ValueError("mix() requires at least one signal")
    return reduce(lambda a, b: a + b, signals)


def normalize(signal: Signal, peak: float = 1.0) -> Signal:
    """Scale a signal so its largest absolute sample equals ``peak``.

    A fully silent signal is returned unchanged (no division by zero).
    """
    if peak < 0:
        raise ValueError(f"peak must be non-negative, got {peak}")
    current = float(np.max(np.abs(signal.samples))) if signal.length else 0.0
    if current == 0.0:
        return Signal(signal.samples, signal.sample_rate)
    return signal * (peak / current)


def clip(signal: Signal, threshold: float = 1.0) -> Signal:
    """Hard-clip samples to ``[-threshold, threshold]``."""
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    clipped = np.clip(signal.samples, -threshold, threshold)
    return Signal(clipped, signal.sample_rate)


def pan(signal: Signal, position: float) -> Signal:
    """Pan a signal across the stereo field using equal-power panning.

    Parameters
    ----------
    signal:
        Input signal; stereo input is down-mixed to mono first.
    position:
        Pan position in ``[-1, 1]``: ``-1`` hard left, ``0`` centre, ``+1``
        hard right. Centre applies a ~0.707 gain to each side (equal power).

    Returns
    -------
    Signal
        A stereo signal.
    """
    if not -1.0 <= position <= 1.0:
        raise ValueError(f"position must be in [-1, 1], got {position}")
    mono = signal.to_mono().samples
    angle = (position + 1.0) * 0.25 * np.pi  # 0 → π/2
    left = np.cos(angle) * mono
    right = np.sin(angle) * mono
    return Signal(np.stack([left, right], axis=1), signal.sample_rate)


def _fade_ramp(length: int, n: int, ascending: bool) -> npt.NDArray[np.float64]:
    """Build a fade gain curve of ``length`` samples ramping over ``n`` samples."""
    curve = np.ones(length, dtype=np.float64)
    n = min(n, length)
    if n <= 0:
        return curve
    ramp = np.linspace(0.0, 1.0, n, endpoint=True)
    if ascending:
        curve[:n] = ramp
    else:
        curve[length - n :] = ramp[::-1]
    return curve


def fade_in(signal: Signal, duration: float) -> Signal:
    """Apply a linear fade-in over the first ``duration`` seconds."""
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    n = int(round(duration * signal.sample_rate))
    curve = _fade_ramp(signal.length, n, ascending=True)
    return _apply_curve(signal, curve)


def fade_out(signal: Signal, duration: float) -> Signal:
    """Apply a linear fade-out over the last ``duration`` seconds."""
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    n = int(round(duration * signal.sample_rate))
    curve = _fade_ramp(signal.length, n, ascending=False)
    return _apply_curve(signal, curve)


def _apply_curve(signal: Signal, curve: npt.NDArray[np.float64]) -> Signal:
    """Multiply a per-sample mono ``curve`` into a (mono or stereo) signal."""
    if signal.channels == 1:
        return Signal(signal.samples * curve, signal.sample_rate)
    return Signal(signal.samples * curve[:, np.newaxis], signal.sample_rate)
