from __future__ import annotations

import array
import queue
import threading
import time
from typing import Final

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ────────────────────────────────── CONFIG ───────────────────────────────────
MODEL_SIZE: Final[str] = "tiny.en"    # fastest English‑only model (<40 MB int8)
DEVICE: Final[str] = "cpu"            # Pi 5 has no CUDA, Metal, etc.
COMPUTE_TYPE: Final[str] = "int8"     # int8 quantization: fastest & smallest
SAMPLE_RATE: Final[int] = 16_000      # Whisper expects 16 kHz mono
CHUNK_SECONDS: Final[float] = 1.0     # lower = snappier; higher = fewer calls
RING_SECONDS: Final[int] = 4          # keep last N seconds in RAM

VAD_PADDING_MS: Final[int] = 150      # speech padding for VAD filter
BEAM_SIZE: Final[int] = 1             # greedy decoding – speed > accuracy

# ─────────────────────────────── GLOBALS ────────────────────────────────────
audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=RING_SECONDS * 2)
stop_event = threading.Event()

CHUNK_SAMPLES: Final[int] = int(SAMPLE_RATE * CHUNK_SECONDS)
MAX_SAMPLES: Final[int] = int(SAMPLE_RATE * RING_SECONDS)

# ─────────────────────────── AUDIO CAPTURE ──────────────────────────────────

def _audio_callback(indata, frames, _time, status):
    """Convert float32 −1…1 mic audio to int16 PCM and push to queue."""
    if status:
        print(status, flush=True)
    pcm16 = (indata[:, 0] * 32768).astype(np.int16)
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        pass  # drop chunk if the CPU is too busy

# ──────────────────────────── TRANSCRIPTION ─────────────────────────────────

def _transcriber(model: WhisperModel):
    ring = array.array("h")  # 16‑bit mono rolling buffer

    while not stop_event.is_set():
        try:
            chunk = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        ring.extend(chunk)
        # Trim to last MAX_SAMPLES samples
        if len(ring) > MAX_SAMPLES:
            del ring[:-MAX_SAMPLES]

        if len(ring) >= CHUNK_SAMPLES:
            pcm = np.frombuffer(ring, dtype=np.int16)[-CHUNK_SAMPLES:]
            audio_f32 = pcm.astype(np.float32) / 32768.0
            segments, _ = model.transcribe(
                audio_f32,
                language="en",
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(speech_pad_ms=VAD_PADDING_MS),
                word_timestamps=False,
            )
            for s in segments:
                print(s.text.strip(), flush=True)

# ──────────────────────────────── MAIN ─────────────────────────────────────

def main() -> None:
    print("Loading faster‑whisper…", flush=True)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Listening…  (Ctrl‑C to quit)", flush=True)

    trans_thread = threading.Thread(target=_transcriber, args=(model,), daemon=True)
    trans_thread.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=0,  # let PortAudio pick low‑latency size
            callback=_audio_callback,
        ):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping…", flush=True)
    finally:
        stop_event.set()
        trans_thread.join()


if __name__ == "__main__":
    main()