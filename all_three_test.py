import requests
import json
import time
import os
import re
import subprocess
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
#Vosk
import wave
import json
from vosk import Model, KaldiRecognizer

#Faster Whisper
import whisper
from faster_whisper import WhisperModel


class LLM_Joke(object):

    def __init__(self, joke_script, extra_info=""):
        self.joke_script = joke_script
        self.extra_info = extra_info
        self.stt_result = ""
        self.url = "http://localhost:11434/api/chat"


    def llama3(self, url, prompt):
        data = {
            "model": "qwen3:1.7b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt

                }
            ],
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['message']['content']

    def faster_whisper_stt(self):
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe("location.wav", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        return text

    def vosk_stt(self, audio_path="location.wav", model_path="vosk-model"):
        # Load Vosk model
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)  # Sample rate expected by Vosk

        # Open the WAV file
        with wave.open(audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                subprocess.run("ffmpeg -i location.wav -ac 1 -ar 16000 -acodec pcm_s16le location2.wav", shell=True, check=True)
                raise ValueError("Audio file must be WAV format, Mono, 16-bit PCM, 16kHz")
            
            while True:
                data = wf.readframes(4000)  # Read 4000 frames at a time
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)

        # Get the transcription result
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")

    def piper_tts(self, location, response, filename):
        command = f"echo \"How we doing, {location}? I hear it's {response} here this time of year.\" | piper --model en_US-lessac-medium --output_file {filename}"
        subprocess.run(command, shell=True, check=True)
        return filename

    def record_audio(self, duration=2, filename="location2.wav", sample_rate=44100):
        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        write(filename, sample_rate, audio)
        subprocess.run("ffmpeg -i location2.wav -ac 1 -ar 16000 -acodec pcm_s16le location.wav", shell=True, check=True)
        print(f"Recording saved as {filename}")

    def main(self):
        self.record_audio()
        start = time.time()
        self.stt_result = self.faster_whisper_stt()
        print(f"Whisper Time: {time.time() - start}")
        #os.remove("location.wav")
        
        start_llm = time.time()
        request = self.joke_script.format(location=self.stt_result)
        response = self.llama3(self.url, request)
        print(f"LLM Time: {time.time() - start_llm}")

        start_tts = time.time()
        response_final = re.split(r'\W+', response)
        output = "response.wav"
        #self.piper_tts(location, response_final[0], output)
        print(f"TTS Time: {time.time() - start_tts}")
        
        print(f"Total Time: {time.time() - start}")
        print(request)
        print(response)
        return(response_final[0])
    
if __name__ == "__main__":
    joke_string = "Based on the likely actual weather in {location} in the Winter, say an option closest to the likely weather (Sunny, Cold, Rainy, Stormy, Overcast). SAY ONLY ONE WORD"
    joke = LLM_Joke(joke_string)
    joke.main()
    os.remove("location.wav")
