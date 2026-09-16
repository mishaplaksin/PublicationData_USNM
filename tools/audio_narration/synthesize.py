#!/usr/bin/env python3
"""Turn a speaker-tagged narration script into a single MP3.

Reads the lightweight script format produced for these narrations:

    [HOST] Two numbers. Hold on to both of them.
    [SKEPTIC] One session. Ninety days.
    [PAUSE 800]
    [[Chapter name]]
    # a comment

Each speaker tag maps to a Kokoro voice (see VOICES). Lines are split into
sentence-sized chunks so the TTS model stays in its comfortable length range,
synthesised one chunk at a time, joined with short inter-chunk gaps, and
encoded to MP3 with ffmpeg.

A text normaliser rewrites the symbols and units that a phonemiser mangles
("W/cm²" -> "watts per square centimetre") so the script can keep whichever
form reads better on the page.

Usage:
    python3 synthesize.py scripts/arise_v7_narration.txt -o out/arise_v7.mp3

Requires: kokoro-onnx, soundfile, numpy, espeak-ng, ffmpeg, and the Kokoro
model files (see setup.sh).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "kokoro-v1.0.onnx"
DEFAULT_VOICES = HERE / "models" / "voices-v1.0.bin"

# Speaker tag -> (voice, speed). Kokoro ships 50+ voices; these two contrast
# well over car speakers (British male narrator vs American female questioner).
VOICES = {
    "HOST": ("bm_fable", 1.0),
    "SKEPTIC": ("af_heart", 1.02),
}

SAMPLE_RATE = 24000

# Gap inserted after each chunk of a line, and after a whole line, in seconds.
GAP_WITHIN_LINE = 0.10
GAP_AFTER_LINE = 0.34
# Extra breathing room when the speaker changes.
GAP_SPEAKER_CHANGE = 0.26

MAX_CHUNK_CHARS = 320

# --------------------------------------------------------------------------
# Text normalisation: make units and symbols survive phonemisation.
# --------------------------------------------------------------------------

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

UNIT_REPLACEMENTS = [
    (r"\bW/cm²", " watts per square centimetre"),
    (r"\bmN/m\b", " millinewtons per metre"),
    (r"\bN/m\b", " newtons per metre"),
    (r"\bMPa\b", " megapascals"),
    (r"\bkPa\b", " kilopascals"),
    (r"\bkHz\b", " kilohertz"),
    (r"\bMHz\b", " megahertz"),
    (r"\bHz\b", " hertz"),
    (r"\bms\b", " milliseconds"),
    (r"\bNp/cm\b", " nepers per centimetre"),
    (r"\bdB/cm\b", " decibels per centimetre"),
    (r"\bmm\b", " millimetres"),
    (r"\bcm²", " square centimetres"),
    (r"\bISPPA\b", "I-S-P-P-A"),
    (r"\bMI\b", "mechanical index"),
    (r"\bTUS\b", "T-U-S"),
    (r"\bTMS\b", "T-M-S"),
    (r"\bDBS\b", "D-B-S"),
    (r"\bMSCs?\b", "mechanosensitive channels"),
    (r"\bNAc\b", "nucleus accumbens"),
    (r"\bMSNs?\b", "medium spiny neurons"),
    (r"\bAFM\b", "A-F-M"),
    (r"\bMR-ARFI\b", "M-R A-R-F-I"),
    (r"\bITRUSST\b", "eye-TRUSST"),
    (r"\bLTP\b", "L-T-P"),
    (r"\bLTD\b", "L-T-D"),
    (r"\bBDNF\b", "B-D-N-F"),
    (r"\bNMDA\b", "N-M-D-A"),
    (r"\bAMPA\b", "AM-pa"),
    (r"\bCREB\b", "creb"),
    (r"\bCaMKII\b", "cam-kinase two"),
    (r"\bCaMKIV\b", "cam-kinase four"),
    (r"\bSNARE\b", "snare"),
    (r"\bATP\b", "A-T-P"),
    (r"\bGABA-A\b", "GABA-A"),
    (r"\bGABAA\b", "GABA-A"),
    (r"\bTrkB\b", "track-B"),
    (r"\bEMG\b", "E-M-G"),
    (r"\bFRET\b", "fret"),
    (r"\bOUD\b", "opioid use disorder"),
    (r"\bSUD\b", "substance use disorder"),
]

CHAR_REPLACEMENTS = [
    ("—", ", "),
    ("–", " to "),
    ("‑", "-"),
    ("×", " times "),
    ("≈", " approximately "),
    ("≲", " at most about "),
    ("≳", " at least about "),
    ("≤", " at most "),
    ("≥", " at least "),
    ("<", " under "),
    (">", " over "),
    ("°C", " degrees Celsius"),
    ("µ", "micro"),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
    ("…", ", "),
]


def normalise(text: str) -> str:
    """Rewrite symbols and unit abbreviations into speakable words."""
    text = text.translate(SUPERSCRIPTS)
    for pattern, replacement in UNIT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    for old, new in CHAR_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


# --------------------------------------------------------------------------
# Script parsing
# --------------------------------------------------------------------------

LINE_RE = re.compile(r"^\[([A-Z]+)\]\s*(.+)$")
PAUSE_RE = re.compile(r"^\[PAUSE\s+(\d+)\]$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\[\[(.+)\]\]$")


class Speech:
    __slots__ = ("speaker", "text")

    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text


class Pause:
    __slots__ = ("ms",)

    def __init__(self, ms: int):
        self.ms = ms


class Chapter:
    __slots__ = ("title",)

    def __init__(self, title: str):
        self.title = title


def parse_script(path: Path) -> list:
    events: list = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if m := CHAPTER_RE.match(line):
            events.append(Chapter(m.group(1).strip()))
            continue
        if m := PAUSE_RE.match(line):
            events.append(Pause(int(m.group(1))))
            continue
        if m := LINE_RE.match(line):
            speaker, text = m.group(1), m.group(2).strip()
            if speaker not in VOICES:
                raise SystemExit(
                    f"{path}:{lineno}: unknown speaker [{speaker}]; "
                    f"known: {', '.join(sorted(VOICES))}"
                )
            events.append(Speech(speaker, text))
            continue

        raise SystemExit(f"{path}:{lineno}: cannot parse line: {raw!r}")
    return events


def split_for_tts(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split into sentence-ish chunks no longer than `limit` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > limit:
            # Very long sentence: break it at clause boundaries.
            parts = re.split(r"(?<=[,;:])\s+", sentence)
        else:
            parts = [sentence]
        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
    if current:
        chunks.append(current)
    return chunks or [text]


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def synthesize(events: list, model: Path, voices: Path) -> tuple[np.ndarray, list]:
    from kokoro_onnx import Kokoro

    print(f"loading model {model.name} ...", flush=True)
    kokoro = Kokoro(str(model), str(voices))

    segments: list[np.ndarray] = []
    chapters: list[tuple[str, float]] = []
    elapsed = 0.0
    previous_speaker: str | None = None
    spoken = sum(1 for e in events if isinstance(e, Speech))
    done = 0
    started = time.time()

    def emit(audio: np.ndarray) -> None:
        nonlocal elapsed
        segments.append(audio)
        elapsed += len(audio) / SAMPLE_RATE

    for event in events:
        if isinstance(event, Chapter):
            chapters.append((event.title, elapsed))
            print(f"  [{fmt_duration(elapsed)}] {event.title}", flush=True)
            continue

        if isinstance(event, Pause):
            emit(silence(event.ms / 1000.0))
            continue

        voice, speed = VOICES[event.speaker]
        if previous_speaker is not None and previous_speaker != event.speaker:
            emit(silence(GAP_SPEAKER_CHANGE))
        previous_speaker = event.speaker

        text = normalise(event.text)
        for i, chunk in enumerate(split_for_tts(text)):
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
            if sr != SAMPLE_RATE:
                raise SystemExit(f"unexpected sample rate from model: {sr}")
            emit(np.asarray(samples, dtype=np.float32))
            emit(silence(GAP_WITHIN_LINE))
        emit(silence(GAP_AFTER_LINE - GAP_WITHIN_LINE))

        done += 1
        if done % 10 == 0 or done == spoken:
            rate = elapsed / max(time.time() - started, 1e-6)
            print(f"  {done}/{spoken} lines -> {fmt_duration(elapsed)} audio "
                  f"({rate:.1f}x realtime)", flush=True)

    return np.concatenate(segments), chapters


def encode_mp3(audio: np.ndarray, out: Path, title: str, artist: str,
               bitrate: str) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH; install it (see setup.sh)")

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak) * 0.97  # headroom before the loudness filter

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "narration.wav"
        sf.write(str(wav), audio, SAMPLE_RATE)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav),
            # Speech-friendly mastering for a noisy car cabin: gentle high-pass
            # to drop rumble, then EBU R128 loudness normalisation so the whole
            # file sits at a consistent, audible level.
            "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame", "-b:a", bitrate, "-ar", "44100", "-ac", "1",
            "-metadata", f"title={title}",
            "-metadata", f"artist={artist}",
            "-metadata", "genre=Speech",
            str(out),
        ]
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", type=Path, help="speaker-tagged narration script")
    ap.add_argument("-o", "--output", type=Path, default=Path("out/narration.mp3"))
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--voices", type=Path, default=DEFAULT_VOICES)
    ap.add_argument("--bitrate", default="96k", help="MP3 bitrate (default 96k mono)")
    ap.add_argument("--title", default="ARISE — audio narration")
    ap.add_argument("--artist", default="Insightec / ARISE review")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse, normalise and report word counts without synthesising")
    args = ap.parse_args(argv)

    if not args.script.is_file():
        print(f"error: no such script: {args.script}", file=sys.stderr)
        return 1

    events = parse_script(args.script)
    words = sum(len(normalise(e.text).split()) for e in events if isinstance(e, Speech))
    lines = sum(1 for e in events if isinstance(e, Speech))
    print(f"{args.script}: {lines} spoken lines, {words:,} words "
          f"(~{words / 155:.0f} min at 155 wpm)")

    if args.dry_run:
        for event in events:
            if isinstance(event, Speech):
                print(f"[{event.speaker}] {normalise(event.text)}")
            elif isinstance(event, Chapter):
                print(f"\n=== {event.title} ===")
        return 0

    for path, what in ((args.model, "model"), (args.voices, "voice pack")):
        if not path.is_file():
            print(f"error: missing {what}: {path}\nrun setup.sh first",
                  file=sys.stderr)
            return 1

    audio, chapters = synthesize(events, args.model, args.voices)
    duration = len(audio) / SAMPLE_RATE
    print(f"encoding {fmt_duration(duration)} of audio -> {args.output}")
    encode_mp3(audio, args.output, args.title, args.artist, args.bitrate)

    size_mb = args.output.stat().st_size / 1e6
    print(f"done: {args.output} ({fmt_duration(duration)}, {size_mb:.1f} MB)")
    if chapters:
        print("\nchapters:")
        for title, at in chapters:
            print(f"  {fmt_duration(at):>6}  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
