"""The :class:`Signal` primitive: a NumPy audio buffer plus its sample rate.

``Signal`` is the substrate every other layer is built on. It is treated as
immutable: every operation returns a *new* ``Signal`` and never mutates the
array handed to the constructor or the caller's arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["Signal"]


class Signal:
    """An immutable-ish audio buffer.

    Parameters
    ----------
    samples:
        Sample data. Shape ``(n,)`` for mono or ``(n, 2)`` for stereo. The
        array is copied and cast to ``float64`` so the caller's array is never
        mutated.
    sample_rate:
        Sampling rate in Hz. Must be a positive integer.

    Attributes
    ----------
    samples:
        The underlying ``float64`` NumPy array.
    sample_rate:
        Sampling rate in Hz.
    """

    samples: npt.NDArray[np.float64]
    sample_rate: int

    def __init__(self, samples: npt.ArrayLike, sample_rate: int) -> None:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim not in (1, 2):
            raise ValueError(f"samples must be 1-D (mono) or 2-D (stereo), got {arr.ndim}-D")
        if arr.ndim == 2 and arr.shape[1] not in (1, 2):
            raise ValueError(f"stereo samples must have 1 or 2 channels, got {arr.shape[1]}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        # Copy so we own the data and the caller's array is never mutated.
        self.samples = arr.copy()
        self.sample_rate = sample_rate

    @property
    def length(self) -> int:
        """Number of samples (frames) in the signal."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.length / self.sample_rate

    @property
    def channels(self) -> int:
        """Number of channels: ``1`` for mono, ``2`` for stereo."""
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"Signal(length={self.length}, channels={self.channels}, "
            f"sample_rate={self.sample_rate})"
        )

    # -- channel conversion ------------------------------------------------

    def to_mono(self) -> Signal:
        """Return a mono copy: stereo is averaged across channels."""
        if self.channels == 1:
            return Signal(self.samples.reshape(-1), self.sample_rate)
        return Signal(self.samples.mean(axis=1), self.sample_rate)

    def to_stereo(self) -> Signal:
        """Return a stereo copy: mono is duplicated to both channels."""
        if self.channels == 2:
            return Signal(self.samples, self.sample_rate)
        mono = self.samples.reshape(-1)
        return Signal(np.stack([mono, mono], axis=1), self.sample_rate)

    # -- shaping -----------------------------------------------------------

    def pad_to(self, n: int) -> Signal:
        """Return a copy zero-padded at the tail to at least ``n`` samples.

        If the signal already has ``>= n`` samples it is returned unchanged
        (as a copy); ``pad_to`` only extends, never truncates.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        pad = n - self.length
        if pad <= 0:
            return Signal(self.samples, self.sample_rate)
        width = ((0, pad), (0, 0)) if self.samples.ndim == 2 else (0, pad)
        return Signal(np.pad(self.samples, width), self.sample_rate)

    def slice(self, start_s: float, end_s: float) -> Signal:
        """Return the ``[start_s, end_s)`` time slice as a new signal.

        Bounds are clamped to the signal's extent; ``start_s`` must be
        non-negative and ``end_s >= start_s``.
        """
        if start_s < 0:
            raise ValueError(f"start_s must be non-negative, got {start_s}")
        if end_s < start_s:
            raise ValueError(f"end_s ({end_s}) must be >= start_s ({start_s})")
        i = min(int(round(start_s * self.sample_rate)), self.length)
        j = min(int(round(end_s * self.sample_rate)), self.length)
        return Signal(self.samples[i:j], self.sample_rate)

    # -- arithmetic --------------------------------------------------------

    def __mul__(self, factor: float) -> Signal:
        """Scale amplitude by a scalar ``factor`` (linear gain)."""
        if not isinstance(factor, (int, float)):
            return NotImplemented
        return Signal(self.samples * float(factor), self.sample_rate)

    __rmul__ = __mul__

    def __add__(self, other: Signal) -> Signal:
        """Mix two signals, aligning length (zero-pad) and channels (up-mix).

        The two signals must share a sample rate. If either is stereo the
        result is stereo; the shorter signal is zero-padded to the longer.
        """
        if not isinstance(other, Signal):
            return NotImplemented
        if other.sample_rate != self.sample_rate:
            raise ValueError(
                f"cannot mix signals with different sample rates "
                f"({self.sample_rate} vs {other.sample_rate})"
            )
        a, b = self, other
        if a.channels == 2 or b.channels == 2:
            a, b = a.to_stereo(), b.to_stereo()
        n = max(a.length, b.length)
        return Signal(a.pad_to(n).samples + b.pad_to(n).samples, self.sample_rate)
