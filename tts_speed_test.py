#!/usr/bin/env python3
"""
tts_make_wav.py  –  Tiny wrapper around Coqui-TTS that speaks a sentence to a WAV.

Example:
    python tts_make_wav.py "Hello world" \
        --model tts_models/en/ljspeech/fast_pitch \
        --speaker default --lang en --out hello.wav
"""

import argparse
import os
import warnings
from pathlib import Path

# Silence warnings / tqdm bars for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["COQUI_TQDM_SILENT"] = "1"

try:
    from TTS.api import TTS
except ModuleNotFoundError:
    raise SystemExit("❗  Install first:  pip install -U coqui-tts")

# -------- helpers ------------------------------------------------------------

def make_wav(text: str,
             model_name: str,
             outfile: Path,
             speaker: str | None = None,
             lang: str | None = None) -> None:
    """
    Synthesize *text* with *model_name* and write 16-bit PCM to *outfile*.
    Works with current Coqui-TTS where TTS.save_wav() was removed.
    """
    print(f"Loading model: {model_name}")
    # 'progress_bar=False' hides tqdm model-download bars (optional)
    tts = TTS(model_name, progress_bar=False)

    print(f"Synthesizing → {outfile}")
    # Current API: single call that writes the file for us
    tts.tts_to_file(text=text,
                    speaker=speaker,
                    language=lang,
                    file_path=str(outfile))

    print("Done.")


# -------- CLI ----------------------------------------------------------------

def main() -> None:
    fast_default = "tts_models/en/ljspeech/fast_pitch"

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Sentence to speak (use quotes)")
    parser.add_argument("-m", "--model", default=fast_default,
                        help=f"TTS model name (default = {fast_default})")
    parser.add_argument("--speaker", default=None, help="Speaker ID / name")
    parser.add_argument("--lang",    default=None, help="Language code")
    parser.add_argument("--out",     default="output.wav", help="Output WAV filename")
    args = parser.parse_args()

    make_wav(args.text, args.model, Path(args.out), args.speaker, args.lang)


if __name__ == "__main__":
    main()