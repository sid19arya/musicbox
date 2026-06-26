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
