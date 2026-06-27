"""Mixer: sum track signals with per-track gain/pan and master limiting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from codaw.core import dsp
from codaw.core.signal import Signal

__all__ = ["Mixer"]


class Mixer:
    """Sums signals with per-source gain and pan into a master signal.

    Parameters
    ----------
    master_gain:
        Linear gain applied to the summed mix.
    ceiling:
        Peak ceiling in ``(0, 1]``. If the master exceeds it, the whole mix is
        scaled down to the ceiling (a simple brick-wall limiter) so the output
        never hard-clips. Quieter mixes pass through untouched.
    """

    def __init__(self, master_gain: float = 1.0, ceiling: float = 0.99) -> None:
        if master_gain < 0:
            raise ValueError(f"master_gain must be non-negative, got {master_gain}")
        if not 0.0 < ceiling <= 1.0:
            raise ValueError(f"ceiling must be in (0, 1], got {ceiling}")
        self.master_gain = master_gain
        self.ceiling = ceiling

    def mix(
        self,
        signals: Sequence[Signal],
        gains: Sequence[float] | None = None,
        pans: Sequence[float] | None = None,
    ) -> Signal:
        """Mix ``signals`` with optional per-signal ``gains`` and ``pans``.

        Parameters
        ----------
        signals:
            The track signals to combine. Must be non-empty.
        gains:
            Per-signal linear gains (defaults to 1.0 each).
        pans:
            Per-signal pan positions in ``[-1, 1]`` (defaults to centre). A
            non-zero pan up-mixes that signal to stereo.

        Returns
        -------
        Signal
            The mastered mix.
        """
        if not signals:
            raise ValueError("mix() requires at least one signal")
        if gains is not None and len(gains) != len(signals):
            raise ValueError("gains must match the number of signals")
        if pans is not None and len(pans) != len(signals):
            raise ValueError("pans must match the number of signals")

        processed: list[Signal] = []
        for i, sig in enumerate(signals):
            gain = gains[i] if gains is not None else 1.0
            pan = pans[i] if pans is not None else 0.0
            staged = sig * gain
            if pan != 0.0:
                staged = dsp.pan(staged, pan)
            processed.append(staged)

        master = dsp.mix(*processed) * self.master_gain
        return self._limit(master)

    def _limit(self, signal: Signal) -> Signal:
        """Scale down to the ceiling only if the peak exceeds it."""
        peak = float(np.max(np.abs(signal.samples))) if signal.length else 0.0
        if peak > self.ceiling:
            return signal * (self.ceiling / peak)
        return signal
