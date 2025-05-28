import argparse
import os, warnings, logging
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import List, Tuple

import speech_recognition as sr
try:
    from vosk import SetLogLevel  # noqa: E402
    SetLogLevel(-1)               # -1 = completely silent
except ImportError:
    pass
from vosk import Model, KaldiRecognizer
import whisper
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Mute most library loggers
logging.getLogger().setLevel(logging.ERROR)

# ----------------------------------------------------------------------
def convert_to_wav16k_mono(src: Path) -> Path:
    """Return a 16-kHz mono PCM WAV copy of *src* (uses ffmpeg)."""
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", str(src),
        "-ar", "16000",       # sample rate
        "-ac", "1",           # mono
        "-sample_fmt", "s16", # 16-bit PCM
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("ERROR: ffmpeg not found or failed to convert file.")
    return tmp

def record_mic(duration: int) -> Path:
    """Record mic for *duration* seconds to a 16-kHz mono WAV, return path."""
    dest = Path(__file__).resolve().parent / "test.wav"   # always ./test.wav
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print(f"Recording {duration}s → {dest} …")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.record(source, duration=duration)

    with open(dest, "wb") as f:
        f.write(audio.get_wav_data())

    print("Recording saved.\n")
    return dest

# ----------------------------------------------------------------------
def run_pocketsphinx(wav: Path) -> Tuple[float, str]:
    rec = sr.Recognizer()
    with wave.open(str(wav), "rb") as wf:
        audio = sr.AudioData(wf.readframes(wf.getnframes()),
                             wf.getframerate(),
                             wf.getsampwidth())
    t0 = time.perf_counter()
    text = rec.recognize_sphinx(audio)
    return time.perf_counter() - t0, text


def run_vosk(wav: Path) -> Tuple[float, str]:
    # Expect a model in the working dir or VOSK_MODEL env var
    model_path = os.getenv("VOSK_MODEL", "vosk-model-small-en-us-0.15")
    if not os.path.isdir(model_path):
        raise RuntimeError(f"Vosk model not found at {model_path}")
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    wf = wave.open(str(wav), "rb")
    t0 = time.perf_counter()
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        rec.AcceptWaveform(data)
    text = rec.FinalResult()
    return time.perf_counter() - t0, text


def run_whisper(wav: Path) -> Tuple[float, str]:
    model = whisper.load_model("tiny.en")      # load once, outside timer
    t0 = time.perf_counter()
    out = model.transcribe(str(wav), fp16=False, language="en")
    return time.perf_counter() - t0, out["text"].strip()


def run_faster_whisper(wav: Path) -> Tuple[float, str]:
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    segments, _ = model.transcribe(str(wav), beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments)
    return time.perf_counter() - t0, text


# ----------------------------------------------------------------------
ENGINE_FUNCS = {
    "Vosk":           run_vosk,
    "PocketSphinx":   run_pocketsphinx,
    "Whisper":        run_whisper,
    "Faster-Whisper": run_faster_whisper,
}

# ----------------------------------------------------------------------
def main(record: bool) -> None:
    if record:
        wav = record_mic(10)
        #wav = convert_to_wav16k_mono("test.wav")
    else:
        wav = "test.wav"

    print(f"Benchmarking on: {wav}")
    print("-" * 60)

    results: List[Tuple[str, float, str]] = []

    for name, func in ENGINE_FUNCS.items():
        try:
            elapsed, text = func(wav)
            results.append((name, elapsed, text))
        except ModuleNotFoundError:
            print(f"{name:<16} |  n/a   | (module not installed)")
        except Exception as e:
            print(f"{name:<16} |  n/a   | error: {e}")

    # Pretty-print successful results
    for name, elapsed, text in results:
        print(f"{name:<16} | {elapsed:7.3f} s | {text}")


if __name__ == "__main__":
    main(record=False)