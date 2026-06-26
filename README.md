# codaw — a codified DAW

`codaw` ("coded DAW") is a digital audio workstation where **everything is code**.
There is no GUI: to add a note, an instrument, an effect, or a track, you edit
Python. A song is a script that declaratively builds a `Song` object and renders
it to audio or plays it live.

```python
from codaw import Song

song = Song(bpm=120, sample_rate=44100)
song.save("out.wav")   # render to a WAV file
```

> **Status:** MVP. The thin end-to-end slice — `Signal`, the `sine` oscillator,
> and `Song.render`/`Song.save` — is in place and renders a note to a valid WAV.
> The full component model (oscillators, envelopes, instruments, effects,
> patterns, tracks, mixer, CLI) lands feature-by-feature per the roadmap in
> `codified-daw-design-doc.md` §8.

## Architecture

`codaw` is built in five layers, each depending only on the layers below it and
each independently testable:

| Layer | Package | Contents |
|---|---|---|
| 0 — Primitives | `codaw.core` | `Signal`, oscillators, envelopes, DSP helpers, filters |
| 1 — Music theory | `codaw.music` | notes, scales/chords, timing |
| 2 — Components | `codaw.components` | `Instrument`/`Synth`, `Effect`s, `NoteEvent` |
| 3 — Arrangement | `codaw.arrangement` | `Pattern`, `Track`, `Mixer`, `Song` |
| 4 — Interface | `codaw.interface` | song loader, `codaw` CLI |

## Install (with [uv](https://docs.astral.sh/uv/))

```bash
uv venv
uv pip install -e ".[dev]"
```

This installs the runtime deps (NumPy, SciPy, soundfile) plus the dev toolchain
(pytest, pytest-cov, ruff, mypy). For live playback, also install the optional
`playback` extra (requires PortAudio):

```bash
uv pip install -e ".[dev,playback]"
```

## Write and render your first song

Create `songs/hello.py`:

```python
from codaw import Song

song = Song(bpm=120, sample_rate=44100)

if __name__ == "__main__":
    song.save("hello.wav")
```

Then run it:

```bash
python songs/hello.py
```

## Develop

```bash
ruff check .          # lint
ruff format --check . # format check
mypy codaw            # type check
pytest --cov=codaw --cov-fail-under=85   # tests + coverage gate
```

## License

MIT
