"""Amplitude envelopes.

An :class:`ADSR` envelope shapes a signal's amplitude over time through four
phases — Attack, Decay, Sustain, Release. :meth:`ADSR.apply` builds a per-sample
gain curve the length of the input signal and multiplies it in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from codaw.core.signal import Signal

__all__ = ["ADSR"]


@dataclass(frozen=True)
class ADSR:
    """An Attack–Decay–Sustain–Release amplitude envelope.

    The envelope rises linearly from 0 to 1 over ``attack`` seconds, decays to
    the ``sustain`` level over ``decay`` seconds, holds at ``sustain`` until the
    release phase, then falls to ~0 over the final ``release`` seconds.

    Parameters
    ----------
    attack:
        Attack time in seconds (0 → 1 ramp).
    decay:
        Decay time in seconds (1 → ``sustain`` ramp).
    sustain:
        Sustain level in ``[0, 1]``.
    release:
        Release time in seconds (``sustain`` → 0 ramp), taken from the tail of
        the signal.
    """

    attack: float
    decay: float
    sustain: float
    release: float

    def __post_init__(self) -> None:
        if self.attack < 0 or self.decay < 0 or self.release < 0:
            raise ValueError("attack, decay and release must be non-negative")
        if not 0.0 <= self.sustain <= 1.0:
            raise ValueError(f"sustain must be in [0, 1], got {self.sustain}")

    def envelope(self, length: int, sample_rate: int) -> npt.NDArray[np.float64]:
        """Build the per-sample gain curve for ``length`` samples.

        Parameters
        ----------
        length:
            Number of samples to generate.
        sample_rate:
            Sampling rate in Hz.

        Returns
        -------
        numpy.ndarray
            A ``float64`` gain curve of shape ``(length,)`` in ``[0, 1]``.
        """
        if length <= 0:
            return np.zeros(max(length, 0), dtype=np.float64)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        a = int(round(self.attack * sample_rate))
        d = int(round(self.decay * sample_rate))
        r = int(round(self.release * sample_rate))

        env = np.empty(length, dtype=np.float64)

        # Release is carved from the tail; the A/D/S body fills the rest.
        r = min(r, length)
        body_len = length - r

        idx = 0
        # Attack: 0 -> 1.
        if a > 0 and idx < body_len:
            n = min(a, body_len - idx)
            env[idx : idx + n] = np.linspace(0.0, 1.0, n, endpoint=False)
            idx += n
        # Decay: 1 -> sustain.
        if d > 0 and idx < body_len:
            n = min(d, body_len - idx)
            env[idx : idx + n] = np.linspace(1.0, self.sustain, n, endpoint=False)
            idx += n
        # Sustain: hold. The level reached so far is 1.0 if decay hasn't run,
        # else sustain.
        if idx < body_len:
            level = self.sustain if d > 0 else 1.0
            env[idx:body_len] = level
            idx = body_len

        # Release: from the level at the end of the body down to 0.
        if r > 0:
            start_level = env[body_len - 1] if body_len > 0 else 1.0
            env[body_len:length] = np.linspace(start_level, 0.0, r, endpoint=True)

        return env

    def apply(self, signal: Signal) -> Signal:
        """Apply the envelope to ``signal``, returning a new enveloped signal.

        Parameters
        ----------
        signal:
            The input signal.

        Returns
        -------
        Signal
            A new signal with the per-sample gain curve applied. Stereo signals
            receive the same curve on both channels.
        """
        env = self.envelope(signal.length, signal.sample_rate)
        if signal.channels == 1:
            shaped = signal.samples * env
        else:
            shaped = signal.samples * env[:, np.newaxis]
        return Signal(shaped, signal.sample_rate)
