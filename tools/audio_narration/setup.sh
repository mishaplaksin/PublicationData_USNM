#!/usr/bin/env bash
# Install everything synthesize.py needs: system packages, Python packages and
# the Kokoro TTS model files. Safe to re-run; model downloads are skipped if
# the files are already present and non-trivially sized.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$HERE/models"
RELEASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

say() { printf '\n==> %s\n' "$*"; }

say "system packages (espeak-ng for phonemisation, ffmpeg for MP3)"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq espeak-ng ffmpeg
else
  echo "no apt-get; install espeak-ng and ffmpeg yourself" >&2
fi

say "python packages"
python3 -m pip install --quiet --upgrade \
  python-docx kokoro-onnx soundfile numpy

say "kokoro model files -> $MODELS"
mkdir -p "$MODELS"
fetch() {
  local name="$1" min_bytes="$2" dest="$MODELS/$1"
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" -ge "$min_bytes" ]; then
    echo "  have $name"
    return
  fi
  echo "  downloading $name ..."
  curl -fsSL --retry 4 --retry-delay 2 -o "$dest" "$RELEASE/$name"
}
fetch kokoro-v1.0.onnx 300000000
fetch voices-v1.0.bin 20000000

say "checking"
python3 - <<'PY'
from pathlib import Path
from kokoro_onnx import Kokoro
m = Path(__file__).resolve().parent if False else Path("models")
k = Kokoro(str(m / "kokoro-v1.0.onnx"), str(m / "voices-v1.0.bin"))
print(f"  ok: {len(k.get_voices())} voices available")
PY

say "ready — e.g.:"
echo "  python3 extract_docx.py ARISE_review_v7.docx -o arise_v7.md"
echo "  python3 synthesize.py scripts/arise_v7_narration.txt -o out/arise_v7.mp3"
