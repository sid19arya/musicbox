# Codified DAW — Design Document & Claude Code Execution Brief

> **Purpose of this document.** This is a self-contained brief for a Claude Code instance to build a *codified digital audio workstation* — a DAW where every component is a code module and every song is a script that can be rendered to audio or played live. Read this whole document first, then execute the roadmap in §8 following the workflow rules in §6–§7. Treat the API shapes in §4 and the example song in §4.6 as the contract you are building toward.

---

## 1. Vision & Product Goals

Build a Python library, `codaw` ("coded DAW"), where:

- **Everything is code.** To add a note, an instrument, an effect, or a track, you edit Python. No GUI.
- **Components are modular.** A DAW is decomposed into small, composable units (oscillators, envelopes, instruments, effects, tracks, a mixer). Each is independently usable and testable.
- **Primitives first.** A thin DSP primitive layer (built on NumPy) sits at the bottom. Everything richer is built *on top of* primitives, not bolted on.
- **A song is a script.** A song is a Python module that declaratively constructs a `Song` object and renders or plays it.
- **It renders and it plays.** `song.render("out.wav")` produces a file (the primary, deterministic, CI-friendly path). `song.play()` plays through speakers (best-effort, hardware-dependent, not run in CI).

### Non-goals (v1)
- No GUI / no web frontend.
- No real-time low-latency live performance engine (offline render is the primary path; live playback is "render buffer then play").
- No VST/plugin hosting.
- No notation/MusicXML.

---

## 2. Technology Decisions (already made — do not relitigate)

| Concern | Decision | Rationale |
|---|---|---|
| Language | **Python 3.10+** | Fast iteration, best ecosystem for numeric audio, trivially testable in headless CI. |
| Sample engine | **NumPy** | All audio is `float64`/`float32` NumPy arrays. The substrate for every primitive. |
| Filters / DSP helpers | **SciPy** (`scipy.signal`) | Biquad/Butterworth design, convolution, resampling. Don't hand-roll filter design. |
| File I/O | **soundfile** (libsndfile) | Read/write WAV/FLAC/OGG robustly. |
| Live playback | **sounddevice** (PortAudio) | Cross-platform speaker output. Optional dependency; mocked/skipped in CI. |
| MIDI import (later) | **mido** | Optional, only if/when a MIDI-import feature is scheduled. |
| Analysis (tests only) | **librosa** | Optional test-time helper for pitch/onset assertions. |
| Packaging | **pyproject.toml + hatchling** | Standard, simple. |
| Lint/format | **ruff** (lint + format) | One fast tool. |
| Types | **mypy** | Static checking; the public API must be fully typed. |
| Tests | **pytest** + **pytest-cov** | Deterministic audio assertions. |
| CI | **GitHub Actions** | Lint, type, test matrix, + demo-render artifacts. |
| Git host CLI | **gh** (GitHub CLI) | Create repo, open/merge PRs from the command line. |

**What we build vs. what we lean on:** lean on the libraries above only for solved low-level problems (array math, filter design, file I/O, audio output). Build *ourselves*: oscillators, envelopes, the instrument/effect abstractions, sequencing, tracks, mixer, the `Song` API, the CLI. Those are the modular "components" and are the point of the project.

---

## 3. Architecture — Five Layers

Each layer depends only on layers below it. Each is independently testable. Package layout:

```
codaw/
  __init__.py            # curated public API re-exports
  core/                  # LAYER 0 — primitives
    signal.py            # Signal wrapper around ndarray + sample_rate
    oscillators.py       # sine, saw, square, triangle, noise
    envelopes.py         # ADSR and envelope application
    dsp.py               # gain, mix, normalize, clip, pan, fade
    filters.py           # lowpass/highpass/bandpass via scipy biquads
  music/                 # LAYER 1 — music theory
    pitch.py             # Note: name/MIDI <-> frequency
    scales.py            # scales, chords
    timing.py            # beats <-> seconds given BPM; Duration
  components/            # LAYER 2 — modular DAW components
    instrument.py        # Instrument ABC + Synth concrete impl
    sampler.py           # Sampler (load wav, pitch-shift)  [later feature]
    effects.py           # Effect ABC + Gain, Pan, Delay, Reverb, Filter, Distortion
    events.py            # NoteEvent dataclass
  arrangement/           # LAYER 3 — putting it together
    pattern.py           # Pattern / Clip: a sequence of NoteEvents
    track.py             # Track: instrument + effect chain + clips -> Signal
    mixer.py             # sums tracks with gain/pan
    song.py              # Song/Project: bpm, sample_rate, tracks; render/save/play
  interface/             # LAYER 4 — how users drive it
    loader.py            # exec a song script, return its `song`
    cli.py               # `codaw render song.py out.wav`, `codaw play song.py`
  __main__.py            # entrypoint -> interface.cli
songs/                   # example songs (the "scripts")
tests/
```

### Layer 0 — Primitives (`codaw.core`)
- **`Signal`**: wraps a NumPy array (shape `(n,)` mono or `(n, 2)` stereo) plus `sample_rate`. Methods/properties: `duration`, `channels`, `to_stereo()`, `to_mono()`, `__add__` (mix with length alignment), `__mul__` (scalar gain), `slice(start_s, end_s)`, `pad_to(n)`. Immutable-ish: operations return new `Signal`s. Never mutate caller arrays.
- **Oscillators**: `sine(freq, dur, sr) -> Signal`, plus `saw`, `square`, `triangle`, `noise(seed=...)`. Band-limited is not required for v1, but `noise` MUST accept a seed for determinism.
- **Envelopes**: `ADSR(attack, decay, sustain, release)` with `.apply(signal) -> Signal` producing a per-sample gain envelope. Attack rises 0→1, release falls to ~0.
- **DSP**: `gain`, `mix(*signals)` (length-aligned sum), `normalize(peak=...)`, `clip`, `pan(signal, position)` (mono→stereo, position in [-1,1]), `fade_in/fade_out`.
- **Filters**: `lowpass`, `highpass`, `bandpass` using `scipy.signal` biquad/`sosfilt`. Each takes cutoff(s) and returns a filtered `Signal`.

### Layer 1 — Music theory (`codaw.music`)
- **`Note`**: parse `"C4"`, `"A#3"`, `"Bb5"` and MIDI numbers; `.frequency` (A4=440, 12-TET). Round-trip name↔MIDI↔freq.
- **Scales/Chords**: `Scale("C", "major")` yields notes; `chord("Cmaj7")` helper.
- **Timing**: `beats_to_seconds(beats, bpm)`, `Duration` helpers, note-value constants (`WHOLE`, `HALF`, `QUARTER`, ...).

### Layer 2 — Components (`codaw.components`)
- **`Instrument` (ABC)**: `render(note: Note, duration_s: float, sr: int) -> Signal`. Concrete **`Synth(oscillator, envelope, gain=1.0)`**. This is the primary extension point: a new instrument is a new subclass.
- **`Effect` (ABC)**: `process(signal: Signal) -> Signal`. Concrete: `Gain`, `Pan`, `Delay(time, feedback, mix)`, `Reverb(room_size, mix)` (Schroeder comb+allpass or simple convolution), `Filter(kind, cutoff)`, `Distortion(drive)`. Second primary extension point.
- **`NoteEvent`** dataclass: `pitch: Note`, `start_beat: float`, `duration_beats: float`, `velocity: float = 1.0`.

### Layer 3 — Arrangement (`codaw.arrangement`)
- **`Pattern`**: an ordered collection of `NoteEvent`s, with constructor helpers: `Pattern.from_string("C4:1 E4:1 G4:2")` and `Pattern.sequence(["C4","E4","G4"], dur=0.5)`.
- **`Track`**: holds one `Instrument`, an ordered effect chain (`list[Effect]`), and one or more `Pattern`s. `render(bpm, sr) -> Signal`: render each event via the instrument, place it on a timeline at the correct sample offset, sum, then run the effect chain.
- **`Mixer`**: sums multiple track signals with per-track gain/pan, returns the master `Signal` (with headroom / soft-clip protection).
- **`Song`**: `Song(bpm, sample_rate=44100)`, `.add(track)`, `.render() -> Signal`, `.save(path)`, `.play()`. Rendering must be **deterministic** for a fixed seed.

### Layer 4 — Interface (`codaw.interface`)
- **`loader.load_song(path) -> Song`**: execute a song script in a namespace and return the module-level `song` object (convention: the script assigns `song = Song(...)`).
- **CLI** (`codaw render <script> <out.wav>`, `codaw play <script>`, `codaw render-all songs/ -o build/`).

---

## 4. Public API Contract

### 4.1 Top-level re-exports (`codaw/__init__.py`)
Expose a curated, fully-typed surface: `Song`, `Track`, `Pattern`, `NoteEvent`, `Note`, `Scale`, `Synth`, `Instrument`, `Effect`, oscillator functions, envelope `ADSR`, and the effects (`Gain`, `Pan`, `Delay`, `Reverb`, `Filter`, `Distortion`). A user should be able to `from codaw import *` and build a song.

### 4.6 Target example song (THE CONTRACT — `songs/demo_arpeggio.py`)
The library is "done enough" when a script in approximately this shape renders correctly. Build toward this exact ergonomics:

```python
from codaw import Song, Track, Pattern, Synth, ADSR, Reverb, Delay, oscillators

song = Song(bpm=120, sample_rate=44100)

lead = Track(
    name="lead",
    instrument=Synth(oscillator=oscillators.saw, envelope=ADSR(0.01, 0.1, 0.7, 0.2)),
    effects=[Delay(time=0.25, feedback=0.3, mix=0.25), Reverb(room_size=0.6, mix=0.2)],
)
lead.add(Pattern.from_string("C4:1 E4:1 G4:1 C5:1 G4:1 E4:1"))

bass = Track(
    name="bass",
    instrument=Synth(oscillator=oscillators.sine, envelope=ADSR(0.005, 0.05, 0.9, 0.1)),
)
bass.add(Pattern.from_string("C2:2 C2:2 G2:2 G2:2"))

song.add(lead)
song.add(bass)

if __name__ == "__main__":
    song.save("demo_arpeggio.wav")   # render to file
    # song.play()                    # live playback (not in CI)
```

> If a cleaner API emerges during implementation, you may refine it — but keep it this declarative and this readable, and update this section + all example songs together in the same PR.

---

## 5. Testing Strategy (this is what makes the project rigorous)

All tests run headless with **no audio hardware**. `sounddevice` is mocked or import-guarded; never call real playback in tests.

**Primitive-level (assert on the signal itself):**
- `sine(440, 1.0, 44100)` → length 44100; peak ≈ 1.0; dominant FFT bin within 1 Hz of 440.
- `ADSR.apply` → first sample ≈ 0, rises during attack, last samples decay toward 0.
- `lowpass` at cutoff C → FFT energy above C is attenuated vs. the unfiltered signal.
- `normalize(peak=0.9)` → resulting peak ≈ 0.9.

**Component-level:**
- `Synth.render(Note("A4"), 1.0, sr)` → non-silent, correct length, dominant freq ≈ 440.
- `Delay` → adds energy after the delay offset; `Gain(0.5)` → RMS halves; `Distortion` → increases harmonic content.

**Music theory:**
- `Note("A4").frequency == 440`; name↔MIDI↔freq round-trips for a table of known notes.

**Determinism / golden files:**
- Rendering the same song twice yields byte-identical output (seed all noise).
- A small golden render's SHA-256 (or summary stats: length, peak, RMS) is asserted stable. Regenerating goldens is a deliberate, reviewed step.

**Integration:**
- Each `songs/*.py` loads and renders without error, produces the expected duration (±1 sample), and is non-silent.

**Coverage gate:** fail CI under **85%** line coverage on the `codaw` package.

---

## 6. Git Workflow Rules (follow exactly)

Emulate a disciplined human engineer. **No direct commits to `main` after the MVP.**

1. **Repo setup (once):** ensure `gh auth status` is authenticated. Create the repo: `gh repo create codaw --public --source=. --remote=origin` (or push to an existing one if the user already made it). Set `main` as default.
2. **MVP:** scaffold + the thinnest end-to-end slice (render one sine note to WAV + one passing test + working CI). Commit directly to `main`, push. This is the *only* direct-to-main work.
3. **Every feature thereafter:**
   - `git checkout main && git pull`
   - `git checkout -b feat/<short-name>` (or `fix/`, `chore/`, `ci/`, `docs/`)
   - Implement in small, logically-scoped **Conventional Commits** (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`). Multiple commits per branch is expected and encouraged — show your work.
   - Each feature branch MUST include tests and keep coverage above the gate.
   - `git push -u origin feat/<short-name>`
   - `gh pr create --fill --base main` with a clear title + body (what/why, what's tested). Use the PR template in §9.
   - **Wait for CI to pass.** If red, push fixes to the same branch until green. Never merge red.
   - `gh pr merge --squash --delete-branch`
   - `git checkout main && git pull`
4. **Branch protection (set up in the CI feature):** require the CI status check to pass before merge. Note in the README that this is configured.
5. **Tags/releases:** after a milestone (e.g. effects complete), tag `v0.x.0` and let the release workflow attach demo renders.

---

## 7. CI/CD Pipeline (GitHub Actions)

Create these workflows. They are themselves delivered via a feature branch + PR (see roadmap PR #2).

### `.github/workflows/ci.yml` — on every push & PR
- Matrix: `python-version: ["3.10", "3.11", "3.12"]` on `ubuntu-latest`.
- System deps: `sudo apt-get install -y libsndfile1 libportaudio2`.
- Steps: install (`pip install -e ".[dev]"`) → `ruff check .` → `ruff format --check .` → `mypy codaw` → `pytest --cov=codaw --cov-report=xml --cov-fail-under=85`.
- Upload coverage as an artifact (Codecov optional).

### `.github/workflows/render-demos.yml` — on push to `main`
- Install runtime deps + the package, then `codaw render-all songs/ -o build/`.
- **Upload the rendered `build/*.wav` as workflow artifacts.** Every green build on main thus produces listenable audio — the build artifact of a DAW is *music*. This is a required, deliberate feature, not a nice-to-have.

### `.github/workflows/release.yml` — on tag `v*`
- Build sdist + wheel (`python -m build`), render demos, create a GitHub Release attaching the wheel and the demo WAVs.

**Notes for the implementer:** keep CI fast (cache pip). Guard `sounddevice` import so the package imports cleanly on runners without PortAudio. Tests must never open an audio device.

---

## 8. Feature Roadmap (one PR each, in order)

Execute top-to-bottom. Each row = one branch → commits → PR → squash-merge. Don't start a row until the previous PR is merged and `main` is green.

| # | Branch | Deliverable | Key acceptance |
|---|---|---|---|
| **MVP** | _main_ | Scaffold (`pyproject.toml`, package skeleton, `Signal`, `sine`, `Song.save` rendering a single hard-coded note), one test, README. | `pytest` green locally; one sine renders to a valid WAV. |
| 1 | `ci/pipeline` | `ci.yml` + `render-demos.yml` + ruff/mypy/pytest config; branch-protection note in README. | CI passes on the PR itself. |
| 2 | `feat/oscillators` | `sine, saw, square, triangle, noise(seed)` + tests. | Each waveform: correct length, expected dominant freq, seeded noise reproducible. |
| 3 | `feat/envelopes` | `ADSR` + `.apply` + tests. | Envelope shape assertions pass. |
| 4 | `feat/signal-ops` | Flesh out `Signal` (`__add__`, `__mul__`, slice, pad, stereo) + `dsp.py` (`gain/mix/normalize/clip/pan/fade`) + tests. | Mixing length-aligns; normalize hits target peak; pan produces stereo. |
| 5 | `feat/pitch-timing` | `Note`, `Scale`, `chord`, `beats_to_seconds`, note-value constants + tests. | A4=440; name/MIDI/freq round-trip; beat→second math correct. |
| 6 | `feat/filters` | `lowpass/highpass/bandpass` via scipy + tests. | Spectral attenuation assertions pass. |
| 7 | `feat/instruments` | `Instrument` ABC + `Synth` + `NoteEvent` + tests. | `Synth.render` non-silent, correct length/pitch. |
| 8 | `feat/effects` | `Effect` ABC + `Gain, Pan, Delay, Reverb, Filter, Distortion` + tests. | Each effect changes the signal in the asserted direction. |
| 9 | `feat/arrangement` | `Pattern` (+ `from_string`/`sequence`), `Track` (timeline placement + effect chain), `Mixer` + tests. | Multi-note pattern places events at correct sample offsets; effect chain applies in order. |
| 10 | `feat/song-api` | `Song` (render/save/play), `loader`, deterministic master render + golden test. | Repeated render is byte-identical; golden stats stable. |
| 11 | `feat/cli` | `codaw render`, `codaw play`, `codaw render-all` + `__main__` + tests (mock playback). | CLI renders the demo to WAV in a tmp dir. |
| 12 | `feat/demo-songs` | `songs/demo_arpeggio.py` (the §4.6 contract) + 1–2 more example songs + integration tests + README usage section. | All songs load, render, are non-silent, correct duration. |
| 13 | `feat/sampler` *(stretch)* | `Sampler` instrument (load WAV, pitch-shift via resample) + tests. | Sample plays back at requested pitch. |
| 14 | `feat/midi-import` *(stretch)* | `mido`-based MIDI→`Pattern` importer + tests. | A small `.mid` imports to expected note events. |

After PR #10, tag `v0.1.0`. After PR #12, tag `v0.2.0`.

---

## 9. Conventions & Templates

**Conventional Commit examples:** `feat(oscillators): add band-limited saw`, `test(effects): assert delay adds post-offset energy`, `ci: cache pip and add 3.12 to matrix`, `fix(signal): align lengths when mixing stereo + mono`.

**PR body template:**
```
## What
<one-line summary>

## Why
<link to roadmap row / motivation>

## Changes
- <bullet>
- <bullet>

## Tests
- <what new tests assert; coverage stays > gate>

## Notes
<determinism / follow-ups / anything reviewers should know>
```

**Coding standards:** full type hints on public API; docstrings on every public class/function (numpy style); pure functions in `core` (no global state, no mutation of inputs); no audio device access outside `Song.play` / CLI `play`; seed all randomness.

---

## 10. Prerequisites the human must provide (state these back, then proceed)
- A GitHub account with **`gh` authenticated** (`gh auth login`), able to create repos and PRs.
- Confirmation of repo name/visibility (default: `codaw`, public).
- Awareness that **live playback (`play`) cannot be verified in CI** — only offline rendering is tested automatically; playback is smoke-tested with mocks and must be tried manually.
- Linux/macOS recommended for local playback (`portaudio`); CI runs render-only.

---

## 11. Definition of Done (v1)
- All roadmap PRs #MVP–#12 merged; `main` green.
- `pip install -e ".[dev]"` then `codaw render songs/demo_arpeggio.py out.wav` produces audible, correct music.
- CI: lint + format + mypy + tests (>85% coverage) green on every PR; `render-demos` uploads WAV artifacts on main.
- README documents install, "write your first song," the component model, and how to add a new instrument/effect (i.e., subclass `Instrument`/`Effect`).
- Tagged `v0.2.0` with demo renders attached to the release.

---

# 12. The Prompt to Paste into Claude Code

> Copy everything in this block as your kickoff message to a Claude Code instance (with this document available in the repo or pasted alongside it).

```
You are building `codaw`, a codified DAW in Python, per the attached design document
(codified-daw-design-doc.md). Read the entire document first; it is the contract.

Operating rules:
1. Confirm prerequisites in §10 (gh authenticated, repo name). If gh is not
   authenticated, stop and tell me how to fix it. Otherwise proceed autonomously.
2. Build strictly in the order of the §8 roadmap. One feature = one branch = one PR.
3. After the MVP (committed directly to main with working CI), NEVER commit to main
   again. For every feature: branch -> small Conventional Commits with tests ->
   push -> `gh pr create` -> wait for CI green (fix on the branch if red) ->
   `gh pr merge --squash --delete-branch` -> pull main. Use the §9 PR template.
4. Honor the architecture (§3), API contract (§4, esp. the §4.6 example song), and
   testing strategy (§5). Keep coverage above 85%.
5. Stand up the GitHub Actions pipeline (§7) in PR #1, including the render-demos job
   that uploads rendered WAVs as build artifacts.
6. Keep the public API fully typed and the song-writing ergonomics as clean as §4.6.
7. After each merged PR, post a one-line status: what merged, CI status, what's next.
8. When you hit the Definition of Done (§11), tag v0.2.0, render the demos, and give
   me the repo URL plus instructions to render and play a song locally.

Start now: confirm prerequisites, then scaffold + ship the MVP.
```
