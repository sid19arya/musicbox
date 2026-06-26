"""Effects: the :class:`Effect` extension point and concrete processors.

An effect transforms one :class:`Signal` into another. Effects may change the
length (delay/reverb add a decaying tail) and the channel count (pan makes mono
stereo); downstream mixing length-aligns automatically. New effects are new
:class:`Effect` subclasses.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from codaw.core import dsp, filters
from codaw.core.signal import Signal

__all__ = ["Effect", "Gain", "Pan", "Delay", "Reverb", "Filter", "Distortion"]


class Effect(ABC):
    """Abstract base class for effects.

    A subclass implements :meth:`process`, returning a new :class:`Signal`. This
    is the second primary extension point of the library (after instruments).
    """

    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        """Process ``signal`` and return a new :class:`Signal`."""
        raise NotImplementedError


class Gain(Effect):
    """Scale amplitude by a constant linear ``factor``."""

    def __init__(self, factor: float) -> None:
        if factor < 0:
            raise ValueError(f"factor must be non-negative, got {factor}")
        self.factor = factor

    def process(self, signal: Signal) -> Signal:
        return dsp.gain(signal, self.factor)


class Pan(Effect):
    """Pan a (mono) signal across the stereo field; ``position`` in ``[-1, 1]``."""

    def __init__(self, position: float) -> None:
        if not -1.0 <= position <= 1.0:
            raise ValueError(f"position must be in [-1, 1], got {position}")
        self.position = position

    def process(self, signal: Signal) -> Signal:
        return dsp.pan(signal, self.position)


def _feedback_comb(
    x: npt.NDArray[np.float64], delay: int, feedback: float
) -> npt.NDArray[np.float64]:
    """Apply a feedback comb ``y[i] = x[i] + feedback * y[i-delay]`` along axis 0."""
    b = np.zeros(delay + 1, dtype=np.float64)
    b[0] = 1.0
    a = np.zeros(delay + 1, dtype=np.float64)
    a[0] = 1.0
    a[delay] = -feedback
    return np.asarray(scipy_signal.lfilter(b, a, x, axis=0), dtype=np.float64)


def _delayed(x: npt.NDArray[np.float64], delay: int, feedback: float) -> npt.NDArray[np.float64]:
    """A pure echo line: ``y[i] = x[i-delay] + feedback * y[i-delay]`` (no dry)."""
    b = np.zeros(delay + 1, dtype=np.float64)
    b[delay] = 1.0
    a = np.zeros(delay + 1, dtype=np.float64)
    a[0] = 1.0
    a[delay] = -feedback
    return np.asarray(scipy_signal.lfilter(b, a, x, axis=0), dtype=np.float64)


def _schroeder_allpass(x: npt.NDArray[np.float64], delay: int, g: float) -> npt.NDArray[np.float64]:
    """Schroeder all-pass: ``H(z) = (-g + z^-D) / (1 - g z^-D)``."""
    b = np.zeros(delay + 1, dtype=np.float64)
    b[0] = -g
    b[delay] = 1.0
    a = np.zeros(delay + 1, dtype=np.float64)
    a[0] = 1.0
    a[delay] = -g
    return np.asarray(scipy_signal.lfilter(b, a, x, axis=0), dtype=np.float64)


class Delay(Effect):
    """A feedback delay (echo).

    Parameters
    ----------
    time:
        Delay time in seconds between echoes.
    feedback:
        Echo feedback in ``[0, 1)``: how much of each echo feeds the next.
    mix:
        Wet level in ``[0, 1]`` added to the dry signal.
    """

    def __init__(self, time: float, feedback: float = 0.3, mix: float = 0.25) -> None:
        if time <= 0:
            raise ValueError(f"time must be positive, got {time}")
        if not 0.0 <= feedback < 1.0:
            raise ValueError(f"feedback must be in [0, 1), got {feedback}")
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"mix must be in [0, 1], got {mix}")
        self.time = time
        self.feedback = feedback
        self.mix = mix

    def _tail_samples(self, delay: int) -> int:
        """Number of samples needed for echoes to decay to ~-60 dB."""
        if self.feedback == 0.0:
            return delay
        echoes = min(int(math.ceil(math.log(0.001) / math.log(self.feedback))), 64)
        return delay * max(echoes, 1)

    def process(self, signal: Signal) -> Signal:
        delay = int(round(self.time * signal.sample_rate))
        padded = signal.pad_to(signal.length + self._tail_samples(delay))
        dry = padded.samples
        wet = _delayed(dry, delay, self.feedback)
        return Signal(dry + self.mix * wet, signal.sample_rate)


class Reverb(Effect):
    """A Schroeder reverb: parallel feedback combs into series all-pass filters.

    Parameters
    ----------
    room_size:
        Perceived room size in ``[0, 1]``: larger means a longer, denser tail.
    mix:
        Wet/dry balance in ``[0, 1]`` (0 = dry, 1 = fully wet).
    """

    # Comb and all-pass delay times in seconds (classic Schroeder values).
    _COMB_TIMES = (0.0297, 0.0371, 0.0411, 0.0437)
    _ALLPASS_TIMES = (0.005, 0.0017)

    def __init__(self, room_size: float = 0.5, mix: float = 0.3) -> None:
        if not 0.0 <= room_size <= 1.0:
            raise ValueError(f"room_size must be in [0, 1], got {room_size}")
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"mix must be in [0, 1], got {mix}")
        self.room_size = room_size
        self.mix = mix

    def process(self, signal: Signal) -> Signal:
        sr = signal.sample_rate
        feedback = 0.7 + 0.28 * self.room_size  # 0.70 .. 0.98
        tail = int((0.3 + 1.5 * self.room_size) * sr)
        padded = signal.pad_to(signal.length + tail)
        dry = padded.samples

        wet = np.zeros_like(dry)
        for t in self._COMB_TIMES:
            delay = max(int(round(t * sr)), 1)
            wet += _feedback_comb(dry, delay, feedback) * (1.0 - feedback)
        wet /= len(self._COMB_TIMES)
        for t in self._ALLPASS_TIMES:
            delay = max(int(round(t * sr)), 1)
            wet = _schroeder_allpass(wet, delay, 0.7)

        out = (1.0 - self.mix) * dry + self.mix * wet
        return Signal(out, sr)


class Filter(Effect):
    """A filter effect wrapping the core Butterworth filters.

    Parameters
    ----------
    kind:
        ``"lowpass"``, ``"highpass"`` or ``"bandpass"``.
    cutoff:
        Cutoff in Hz. A single float for low/high-pass; a ``(low, high)`` pair
        for band-pass.
    order:
        Filter order.
    """

    def __init__(
        self,
        kind: str,
        cutoff: float | tuple[float, float],
        order: int = 4,
    ) -> None:
        if kind not in ("lowpass", "highpass", "bandpass"):
            raise ValueError(f"unknown filter kind {kind!r}")
        if kind == "bandpass" and not isinstance(cutoff, tuple):
            raise ValueError("bandpass requires a (low, high) cutoff tuple")
        if kind != "bandpass" and isinstance(cutoff, tuple):
            raise ValueError(f"{kind} requires a single cutoff frequency")
        self.kind = kind
        self.cutoff = cutoff
        self.order = order

    def process(self, signal: Signal) -> Signal:
        if self.kind == "lowpass":
            assert not isinstance(self.cutoff, tuple)
            return filters.lowpass(signal, self.cutoff, self.order)
        if self.kind == "highpass":
            assert not isinstance(self.cutoff, tuple)
            return filters.highpass(signal, self.cutoff, self.order)
        assert isinstance(self.cutoff, tuple)
        low, high = self.cutoff
        return filters.bandpass(signal, low, high, self.order)


class Distortion(Effect):
    """A waveshaping distortion using ``tanh`` saturation.

    Parameters
    ----------
    drive:
        Drive amount (> 0): higher values push harder into saturation, adding
        more harmonics. The output is rescaled to the input's peak so the
        change is timbral rather than a level boost.
    """

    def __init__(self, drive: float = 5.0) -> None:
        if drive <= 0:
            raise ValueError(f"drive must be positive, got {drive}")
        self.drive = drive

    def process(self, signal: Signal) -> Signal:
        x = signal.samples
        shaped = np.tanh(self.drive * x)
        in_peak = float(np.max(np.abs(x))) if signal.length else 0.0
        out_peak = float(np.max(np.abs(shaped))) if signal.length else 0.0
        if out_peak > 0.0:
            shaped = shaped * (in_peak / out_peak)
        return Signal(shaped, signal.sample_rate)
