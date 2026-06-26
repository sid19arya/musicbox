"""Instruments: the :class:`Instrument` extension point and the :class:`Synth`.

An instrument turns a pitched note of some duration into a :class:`Signal`.
``Synth`` is the primary concrete instrument: an oscillator shaped by an ADSR
envelope. New instruments are new :class:`Instrument` subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from codaw.core.envelopes import ADSR
from codaw.core.signal import Signal
from codaw.music.pitch import Note

__all__ = ["Instrument", "Synth", "Oscillator"]

# An oscillator is any callable (freq, duration, sample_rate) -> Signal.
Oscillator = Callable[[float, float, int], Signal]


class Instrument(ABC):
    """Abstract base class for instruments.

    A subclass implements :meth:`render`, turning a note and a duration into a
    :class:`Signal`. This is the primary extension point of the library.
    """

    @abstractmethod
    def render(self, note: Note, duration_s: float, sample_rate: int = 44100) -> Signal:
        """Render ``note`` for ``duration_s`` seconds at ``sample_rate``.

        Parameters
        ----------
        note:
            The pitch to sound.
        duration_s:
            Sounding duration in seconds.
        sample_rate:
            Output sample rate in Hz.

        Returns
        -------
        Signal
            The rendered mono signal.
        """
        raise NotImplementedError


class Synth(Instrument):
    """A subtractive-style synth voice: an oscillator shaped by an ADSR envelope.

    Parameters
    ----------
    oscillator:
        A waveform function ``(freq, duration, sample_rate) -> Signal`` such as
        :func:`codaw.core.oscillators.saw`.
    envelope:
        The amplitude envelope applied to each note. Defaults to a gentle
        plucked shape.
    gain:
        A constant linear gain applied to every note.
    """

    def __init__(
        self,
        oscillator: Oscillator,
        envelope: ADSR | None = None,
        gain: float = 1.0,
    ) -> None:
        if gain < 0:
            raise ValueError(f"gain must be non-negative, got {gain}")
        self.oscillator = oscillator
        self.envelope = envelope if envelope is not None else ADSR(0.01, 0.1, 0.7, 0.1)
        self.gain = gain

    def render(self, note: Note, duration_s: float, sample_rate: int = 44100) -> Signal:
        """Render a note: oscillate at the note's frequency, envelope, gain."""
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        tone = self.oscillator(note.frequency, duration_s, sample_rate)
        shaped = self.envelope.apply(tone)
        return shaped * self.gain
