"""Oscillators: functions that generate periodic :class:`Signal` waveforms.

Every oscillator takes a frequency, a duration in seconds and a sample rate,
and returns a mono :class:`Signal` of peak amplitude ``1.0``. Waveforms are not
band-limited in v1 (naive shapes), which is fine for the offline-render use
case. :func:`noise` is the one stochastic generator and **requires a seed path**
for determinism — rendering must be reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from codaw.core.signal import Signal

__all__ = ["sine", "saw", "square", "triangle", "noise"]


def _time_axis(duration: float, sample_rate: int) -> npt.NDArray[np.float64]:
    """Return the sample time axis (in seconds) for ``duration`` at ``sample_rate``."""
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    n = int(round(duration * sample_rate))
    return np.arange(n, dtype=np.float64) / sample_rate


def _phase(freq: float, duration: float, sample_rate: int) -> npt.NDArray[np.float64]:
    """Return the phase ramp in cycles (``freq * t``); take ``mod 1.0`` for one cycle."""
    t = _time_axis(duration, sample_rate)
    return freq * t


def sine(freq: float, duration: float, sample_rate: int = 44100) -> Signal:
    """Generate a sine wave.

    Parameters
    ----------
    freq:
        Frequency in Hz.
    duration:
        Length in seconds.
    sample_rate:
        Sampling rate in Hz.

    Returns
    -------
    Signal
        A mono signal of peak amplitude ``1.0``.
    """
    t = _time_axis(duration, sample_rate)
    return Signal(np.sin(2.0 * np.pi * freq * t), sample_rate)


def saw(freq: float, duration: float, sample_rate: int = 44100) -> Signal:
    """Generate a rising sawtooth wave ramping from ``-1`` to ``+1`` each cycle.

    Parameters
    ----------
    freq:
        Frequency in Hz.
    duration:
        Length in seconds.
    sample_rate:
        Sampling rate in Hz.

    Returns
    -------
    Signal
        A mono signal of peak amplitude ``1.0``.
    """
    phase = _phase(freq, duration, sample_rate)
    # frac in [0,1); map to [-1, 1).
    frac = np.mod(phase, 1.0)
    return Signal(2.0 * frac - 1.0, sample_rate)


def square(
    freq: float,
    duration: float,
    sample_rate: int = 44100,
    duty: float = 0.5,
) -> Signal:
    """Generate a square wave.

    Parameters
    ----------
    freq:
        Frequency in Hz.
    duration:
        Length in seconds.
    sample_rate:
        Sampling rate in Hz.
    duty:
        Duty cycle in ``(0, 1)`` — fraction of each cycle spent at ``+1``.

    Returns
    -------
    Signal
        A mono signal whose samples are ``+1.0`` or ``-1.0``.
    """
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must be in (0, 1), got {duty}")
    phase = _phase(freq, duration, sample_rate)
    frac = np.mod(phase, 1.0)
    return Signal(np.where(frac < duty, 1.0, -1.0), sample_rate)


def triangle(freq: float, duration: float, sample_rate: int = 44100) -> Signal:
    """Generate a triangle wave oscillating linearly between ``-1`` and ``+1``.

    Parameters
    ----------
    freq:
        Frequency in Hz.
    duration:
        Length in seconds.
    sample_rate:
        Sampling rate in Hz.

    Returns
    -------
    Signal
        A mono signal of peak amplitude ``1.0``.
    """
    phase = _phase(freq, duration, sample_rate)
    frac = np.mod(phase, 1.0)
    # 1 - 4|frac-0.5| is a triangle in [-1, 1] starting at -1, peaking at +1 at
    # mid-cycle, and returning to -1.
    return Signal(1.0 - 4.0 * np.abs(frac - 0.5), sample_rate)


def noise(
    duration: float,
    sample_rate: int = 44100,
    *,
    seed: int = 0,
) -> Signal:
    """Generate white noise in ``[-1, 1]``.

    Parameters
    ----------
    duration:
        Length in seconds.
    sample_rate:
        Sampling rate in Hz.
    seed:
        Seed for the PRNG. Required (keyword-only, defaults to ``0``) so renders
        are deterministic and reproducible.

    Returns
    -------
    Signal
        A mono signal of uniform white noise.
    """
    n = _time_axis(duration, sample_rate).shape[0]
    rng = np.random.default_rng(seed)
    return Signal(rng.uniform(-1.0, 1.0, size=n), sample_rate)
