"""Filters: low/high/band-pass via SciPy Butterworth biquad cascades.

Each function designs a Butterworth filter with :func:`scipy.signal.butter`
(second-order sections) and applies it with :func:`scipy.signal.sosfilt`,
returning a new :class:`Signal`. Stereo signals are filtered per channel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from codaw.core.signal import Signal

__all__ = ["lowpass", "highpass", "bandpass"]


def _nyquist(sample_rate: int) -> float:
    return sample_rate / 2.0


def _apply_sos(sos: npt.NDArray[np.float64], sig: Signal) -> Signal:
    """Apply a second-order-sections filter to a (mono or stereo) signal."""
    filtered = scipy_signal.sosfilt(sos, sig.samples, axis=0)
    return Signal(np.asarray(filtered, dtype=np.float64), sig.sample_rate)


def lowpass(signal: Signal, cutoff: float, order: int = 4) -> Signal:
    """Apply a Butterworth low-pass filter.

    Parameters
    ----------
    signal:
        Input signal.
    cutoff:
        Cutoff frequency in Hz (``0 < cutoff < sample_rate / 2``).
    order:
        Filter order (higher = steeper roll-off).

    Returns
    -------
    Signal
        The filtered signal.
    """
    nyq = _nyquist(signal.sample_rate)
    if not 0.0 < cutoff < nyq:
        raise ValueError(f"cutoff must be in (0, {nyq}) Hz, got {cutoff}")
    sos = scipy_signal.butter(order, cutoff / nyq, btype="low", output="sos")
    return _apply_sos(np.asarray(sos, dtype=np.float64), signal)


def highpass(signal: Signal, cutoff: float, order: int = 4) -> Signal:
    """Apply a Butterworth high-pass filter.

    Parameters
    ----------
    signal:
        Input signal.
    cutoff:
        Cutoff frequency in Hz (``0 < cutoff < sample_rate / 2``).
    order:
        Filter order.

    Returns
    -------
    Signal
        The filtered signal.
    """
    nyq = _nyquist(signal.sample_rate)
    if not 0.0 < cutoff < nyq:
        raise ValueError(f"cutoff must be in (0, {nyq}) Hz, got {cutoff}")
    sos = scipy_signal.butter(order, cutoff / nyq, btype="high", output="sos")
    return _apply_sos(np.asarray(sos, dtype=np.float64), signal)


def bandpass(signal: Signal, low: float, high: float, order: int = 4) -> Signal:
    """Apply a Butterworth band-pass filter passing ``[low, high]`` Hz.

    Parameters
    ----------
    signal:
        Input signal.
    low:
        Lower cutoff in Hz.
    high:
        Upper cutoff in Hz (``low < high < sample_rate / 2``).
    order:
        Filter order (per edge).

    Returns
    -------
    Signal
        The filtered signal.
    """
    nyq = _nyquist(signal.sample_rate)
    if not 0.0 < low < high < nyq:
        raise ValueError(f"require 0 < low < high < {nyq} Hz, got low={low}, high={high}")
    sos = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return _apply_sos(np.asarray(sos, dtype=np.float64), signal)
