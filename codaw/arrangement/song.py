"""The :class:`Song` — the top-level object a user builds and renders.

For the MVP, :meth:`Song.render` produces a single hard-coded sine note so the
end-to-end "construct → render → save WAV" path exists and is tested. Tracks,
mixing and a real arrangement engine arrive in later features.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from codaw.core import oscillators
from codaw.core.signal import Signal

__all__ = ["Song"]

# Frequency of the MVP placeholder note (A4 = 440 Hz).
_MVP_NOTE_FREQ = 440.0
_MVP_NOTE_DURATION = 1.0


class Song:
    """A song: a tempo, a sample rate, and (eventually) a set of tracks.

    Parameters
    ----------
    bpm:
        Tempo in beats per minute.
    sample_rate:
        Output sample rate in Hz.
    """

    def __init__(self, bpm: float, sample_rate: int = 44100) -> None:
        if bpm <= 0:
            raise ValueError(f"bpm must be positive, got {bpm}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.bpm = bpm
        self.sample_rate = sample_rate

    def render(self) -> Signal:
        """Render the song to a single :class:`Signal`.

        MVP behaviour: returns a one-second A4 sine note at the song's sample
        rate. This is replaced by the real arrangement renderer in
        ``feat/song-api``.
        """
        return oscillators.sine(_MVP_NOTE_FREQ, _MVP_NOTE_DURATION, self.sample_rate)

    def save(self, path: str | Path) -> Path:
        """Render the song and write it to ``path`` as a WAV file.

        Parameters
        ----------
        path:
            Destination file path.

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        signal = self.render()
        out = Path(path)
        sf.write(str(out), signal.samples, signal.sample_rate)
        return out
