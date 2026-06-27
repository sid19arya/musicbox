"""Tracks: an instrument, an effect chain, and patterns rendered to a timeline."""

from __future__ import annotations

import numpy as np

from codaw.arrangement.pattern import Pattern
from codaw.components.effects import Effect
from codaw.components.instrument import Instrument
from codaw.core.signal import Signal
from codaw.music.timing import beats_to_seconds

__all__ = ["Track"]


class Track:
    """One instrument, an ordered effect chain, and one or more patterns.

    Rendering places each note event on a sample-accurate timeline (scaled by
    velocity), sums them, then runs the effect chain in order.

    Parameters
    ----------
    name:
        A human-readable track name.
    instrument:
        The instrument used to render every note.
    effects:
        An ordered effect chain applied to the summed track signal.
    gain:
        Per-track linear gain used by the mixer.
    pan:
        Per-track pan position in ``[-1, 1]`` used by the mixer.
    """

    def __init__(
        self,
        name: str,
        instrument: Instrument,
        effects: list[Effect] | None = None,
        *,
        gain: float = 1.0,
        pan: float = 0.0,
    ) -> None:
        if not -1.0 <= pan <= 1.0:
            raise ValueError(f"pan must be in [-1, 1], got {pan}")
        if gain < 0:
            raise ValueError(f"gain must be non-negative, got {gain}")
        self.name = name
        self.instrument = instrument
        self.effects = list(effects) if effects is not None else []
        self.gain = gain
        self.pan = pan
        self.patterns: list[Pattern] = []

    def add(self, pattern: Pattern) -> Track:
        """Add a pattern to the track; returns ``self`` for chaining."""
        self.patterns.append(pattern)
        return self

    def render(self, bpm: float, sample_rate: int = 44100) -> Signal:
        """Render all patterns to a single mono :class:`Signal`, post-effects.

        Parameters
        ----------
        bpm:
            Tempo in beats per minute (sets beat→sample mapping).
        sample_rate:
            Output sample rate in Hz.
        """
        placed: list[tuple[int, Signal]] = []
        total = 0
        for pattern in self.patterns:
            for event in pattern:
                duration_s = beats_to_seconds(event.duration_beats, bpm)
                voice = self.instrument.render(event.pitch, duration_s, sample_rate)
                voice = voice * event.velocity
                offset = int(round(beats_to_seconds(event.start_beat, bpm) * sample_rate))
                placed.append((offset, voice))
                total = max(total, offset + voice.length)

        buffer = np.zeros(total, dtype=np.float64)
        for offset, voice in placed:
            buffer[offset : offset + voice.length] += voice.samples

        out: Signal = Signal(buffer, sample_rate)
        for effect in self.effects:
            out = effect.process(out)
        return out

    def __repr__(self) -> str:
        return (
            f"Track(name={self.name!r}, patterns={len(self.patterns)}, effects={len(self.effects)})"
        )
