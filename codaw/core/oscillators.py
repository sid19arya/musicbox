"""Oscillators: functions that generate periodic :class:`Signal` waveforms.

For the MVP only :func:`sine` is implemented; the remaining waveforms arrive in
the ``feat/oscillators`` feature.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from codaw.core.signal import Signal

__all__ = ["sine"]


def _time_axis(duration: float, sample_rate: int) -> npt.NDArray[np.float64]:
    """Return the sample time axis (in seconds) for ``duration`` at ``sample_rate``."""
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")
    n = int(round(duration * sample_rate))
    return np.arange(n, dtype=np.float64) / sample_rate


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
        A mono signal of peak amplitude ``1.0`` (modulo the first/last sample).
    """
    t = _time_axis(duration, sample_rate)
    samples = np.sin(2.0 * np.pi * freq * t)
    return Signal(samples, sample_rate)
