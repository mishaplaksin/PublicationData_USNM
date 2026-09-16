# Audio narration toolchain

Turns a manuscript into a listenable audio programme — built for listening to
the ARISE review in the car, but the tooling is manuscript-agnostic.

Everything runs offline after setup. No TTS API keys, no per-character billing,
no manuscript text leaving the machine.

```
extract_docx.py  .docx  ->  Markdown (full text, tables, headings)
                              |
                              v  (write a narration script by hand)
scripts/*.txt    speaker-tagged narration script
                              |
                              v
synthesize.py    script  ->  MP3 (multi-voice, loudness-normalised)
```

## Setup

```bash
./setup.sh
```

Installs `espeak-ng` (phonemisation) and `ffmpeg` (MP3 encoding) via apt, the
Python packages via pip, and downloads the two Kokoro model files into
`models/` (~350 MB, from the `kokoro-onnx` GitHub release). Re-running is cheap;
existing model files are left alone.

`models/` is git-ignored — run `setup.sh` on a fresh clone rather than
committing 350 MB of weights.

## Usage

Extract the source document so you can read it while writing a script:

```bash
python3 extract_docx.py ARISE_review_v7.docx -o arise_v7.md
```

Check a script parses, and see exactly what the TTS will be fed, without
spending any compute:

```bash
python3 synthesize.py scripts/arise_v7_narration.txt --dry-run
```

Render:

```bash
python3 synthesize.py scripts/arise_v7_narration.txt -o out/arise_v7.mp3
```

Roughly 5× faster than realtime on CPU, so an hour of audio takes about twelve
minutes. Chapter offsets are printed at the end.

## Script format

Plain text, one directive per line:

| Line | Meaning |
| --- | --- |
| `[HOST] text` | spoken by the HOST voice |
| `[SKEPTIC] text` | spoken by the SKEPTIC voice |
| `[PAUSE 800]` | 800 ms of silence |
| `[[Chapter name]]` | chapter marker — not spoken, reported as a timestamp |
| `# text` | comment |

An unknown speaker tag or an unparseable line is a hard error rather than
something silently dropped, so a typo can't quietly delete a paragraph from the
finished recording.

Speakers map to voices in `VOICES` at the top of `synthesize.py`:

```python
VOICES = {
    "HOST":    ("bm_fable", 1.0),
    "SKEPTIC": ("af_heart", 1.02),
}
```

Kokoro ships 50+ voices (`am_michael`, `bm_george`, `af_bella`, …); list them
with:

```bash
python3 -c "from kokoro_onnx import Kokoro; \
  print(sorted(Kokoro('models/kokoro-v1.0.onnx','models/voices-v1.0.bin').get_voices()))"
```

Add a speaker by adding a key to `VOICES` — the parser picks it up
automatically. Setting both keys to the same voice gives a single-narrator
recording.

## Writing a narration script

The script is written by hand, not generated. What made the ARISE one work:

- **Open with the puzzle, not the abstract.** Two numbers that shouldn't both
  be true buy you the next hour.
- **Speak the numbers.** "thirty-four to one hundred and seventy-two watts per
  square centimetre", not "34–172 W/cm²". The normaliser in `synthesize.py`
  handles units and symbols as a safety net, but digits read aloud by a
  phonemiser are hit-and-miss and a written-out number always wins.
- **Name people instead of citing them.** "Creed, Pascoli and Lüscher, in two
  thousand fifteen" — never "(Creed et al., 2015)".
- **Use the second voice for real objections.** The SKEPTIC exists to voice the
  reviewer's question a beat before the HOST answers it. That's what keeps an
  hour of mechanism listenable; it is not there to say "wow, interesting".
- **Keep paragraphs to one idea.** Each line becomes an audio segment with a
  gap after it, so a line is a breath.
- **Put the self-corrections in the foreground.** In a review whose appendix
  lists its own errors, those are the most interesting minutes in the file.

## Implementation notes

- **Text normalisation** (`normalise()`) rewrites units (`W/cm²`,
  `mN/m`, `MPa`), superscripts, dashes and maths symbols, and spells out
  initialisms (`LTP` → `L-T-P`) that a phonemiser would otherwise try to say as
  words. Extend `UNIT_REPLACEMENTS` / `CHAR_REPLACEMENTS` for a new manuscript.
- **Chunking** splits each line at sentence boundaries, then at clause
  boundaries if a sentence is still over 320 characters. Kokoro degrades on very
  long inputs.
- **Pacing** — 100 ms between chunks, 340 ms after a line, plus 260 ms when the
  speaker changes. Turn-taking sounds natural without dragging.
- **Mastering** — a 70 Hz high-pass to shed rumble, then EBU R128 loudness
  normalisation to −16 LUFS. That's the important one for car listening: it
  keeps the level constant so quiet passages stay audible over road noise
  without the loud ones making you reach for the volume. Encoded mono at 96
  kbps, which is transparent for speech and keeps an hour under 45 MB.
- **File size** — an hour at the 96 kbps default is ~41 MiB, which is over the
  30 MiB limit on some upload and messaging paths. `--bitrate 64k` brings an
  hour to ~27 MiB and is still inaudibly different on speech; go straight to
  64k when the file has to be sent somewhere rather than transcoding a 96k
  render afterwards.

## Files

| Path | |
| --- | --- |
| `setup.sh` | one-shot dependency + model install |
| `extract_docx.py` | `.docx` → Markdown |
| `synthesize.py` | narration script → MP3 |
| `scripts/arise_v7_narration.txt` | narration script for ARISE review v7 |
| `models/` | Kokoro weights (git-ignored, created by `setup.sh`) |
| `out/` | rendered audio (git-ignored) |
