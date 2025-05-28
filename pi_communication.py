import socket
import time
import os
import struct
import subprocess
import numpy as np
from all_three_test import LLM_Joke
from deepface import DeepFace
from picamera2 import Picamera2, Preview

#NAO_IP = '10.42.0.36'       # Actual NAO IP
NAO_IP = '10.42.0.179'      # Computer IP
PI_IP = '10.42.0.1'
PORT = 2033


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PI_IP, PORT)) 
start = time.time()

picam0 = Picamera2(0)
picam1 = Picamera2(1)

def send_message(message, target_ip):
    sock.sendto(message.encode(), (target_ip, PORT))
    print(f"Sent message: {message}")

def send_message_audio(file_path, target_ip):
    try:
        with open(file_path, 'rb') as audio_file:
            sequence_num = 0
            while chunk := audio_file.read(1024):
                packet = struct.pack("!I", sequence_num) + chunk  # Add sequence number (4 bytes)
                sock.sendto(packet, (target_ip, PORT))
                sequence_num += 1

            # Send EOF
            sock.sendto(b'EOF', (target_ip, PORT))  
            print(f"Sent audio file: {file_path}")
            
        os.remove(file_path)  # Optional: Remove file after sending
    except Exception as e:
        print(f"Error sending audio file: {e}")

def receive_message():
    data, addr = sock.recvfrom(1024)
    message = data.decode()
    print(f"Received message: {message} from {addr}")
    return message

def cam_initializer():
    config0 = picam0.create_preview_configuration()
    config1 = picam1.create_preview_configuration()

    picam0.configure(config0)
    picam1.configure(config1)

    picam0.start()
    picam1.start()

def num_faces(fps=15, time_check=2):
    frame_interval = 1.0 / fps
    start_call = time.time()
    faces_arr = []
    while True:
        start_time = time.time()
        total_faces = 0
        
        # Capture a frame
        frame0 = picam0.capture_array()
        frame1 = picam1.capture_array()

        try:
            analysis_results = DeepFace.analyze(frame0, actions=("emotion",), enforce_detection=False)
            if analysis_results:
                total_faces += len(analysis_results)

        except Exception as e:
            print(f"Error during face finding (Camera 0): {e}")
            
        try:
            analysis_results = DeepFace.analyze(frame1, actions=("emotion",), enforce_detection=False)
            if analysis_results:
                total_faces += len(analysis_results)

        except Exception as e:
            print(f"Error during face finding (Camera 1): {e}")
            
        # Amount of faces found in frame from both cameras combines
        faces_arr.append(total_faces)

        # Wait until next frame based on FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)
            
        elapsed_total = time.time() - start_call
        if elapsed_total > time_check:
            break
    
    return np.round(sum(faces_arr)/len(faces_arr))

def dom_emotion(fps=15, time_check=2):
    frame_interval = 1.0 / fps
    start_call = time.time()
    emotion_dict = {'sad':0,'happy':0,'angry':0,'neutral':0,'surprise':0,'disgust':0,'fear':0}
    while True:
        start_time = time.time()
        
        # Capture a frame
        frame0 = picam0.capture_array()
        frame1 = picam1.capture_array()

        try:
            analysis_results = DeepFace.analyze(frame0, actions=("emotion",), enforce_detection=False)
            if analysis_results:
                for analysis in analysis_results:
                    for key, value in analysis['emotion'].items():
                        emotion_dict[key] += value

        except Exception as e:
            print(f"Error during emotion detection (Camera 0): {e}")
            
        try:
            analysis_results = DeepFace.analyze(frame1, actions=("emotion",), enforce_detection=False)
            if analysis_results:
                for analysis in analysis_results:
                    for key, value in analysis['emotion'].items():
                        emotion_dict[key] += value

        except Exception as e:
            print(f"Error during emotion detection (Camera 1): {e}")

        # Wait until next frame based on FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)
        
        elapsed_total = time.time() - start_call
        if elapsed_total > time_check:
            break
    
    dominant_emotion = ''
    max_val = 0
    for key, value in emotion_dict:
        if value > max_val:
            max_val = value
            dominant_emotion = key
            
    return dominant_emotion

def make_joke(joke, response, style):
    filename = "response.wav"
    if style == 1:
        speech = f"How we doing, {joke.stt_result}? I hear it's {response} here this time of year."
    if style == 2:
        if response == "high":
            speech = f"This audience member has no idea what I’m talking about. Because with a {joke.stt_result}’s pay, ladies are always wanting to flash their circuit boards."
        if response == "low":
            speech = f"This audience member knows what I’m talking about! No one wants to show their circuit boards to someone with a {joke.stt_result}’s pay. Me and you, buddy, me and you…"

    return speech


def piper_tts(speech, filename):
    #command = f"echo \"{speech}\" | piper --model en_US-lessac-medium --output_file {filename}"
    command = f"echo \"{speech}\" | piper --model ~/comedy-robot-strategies/piper/joey7999.onnx --output_file {filename}"
    subprocess.run(command, shell=True, check=True)
    return filename
    

def execute_command(message):
    message_arr = message.split(" ")
    if message == "start count":
        print("Starting process")
        num_people = num_faces(fps=15, time_check=2)
        print("{} people seen".format(num_people))
        if num_people >= 10:
            send_message("1", NAO_IP)
        elif num_people >= 5:
            send_message("0", NAO_IP)
        else:
            send_message("-1", NAO_IP)
    
    if message == "start emotion":
        print("Starting process")
        dominant_emotion = dom_emotion(fps=15, time_check=2)
        print("{} is dominant emotion".format(num_people))
        if dominant_emotion in ('happy', 'surprise', 'fear'):
            send_message("1", NAO_IP)
        elif dominant_emotion in ('neutral'):
            send_message("0", NAO_IP)
        else:
            send_message("-1", NAO_IP)

    if len(message_arr) > 2:
        print("Starting process")
        if message_arr[2] == "1":
            joke_script = "Based on the likely actual weather in {location} in the Winter, say an option closest to the likely weather (Sunny, Cold, Rainy, Stormy, Overcast, Warm, Windy). SAY ONLY ONE WORD"
            joke = LLM_Joke(joke_script=joke_script)
            response_text = joke.main().lower()
            if response_text in ("sunny", "warm"):
                send_message("0", NAO_IP)
            elif response_text in ("cloudy", "overcast"):
                send_message("1", NAO_IP)
            elif response_text in ("windy"):
                send_message("2", NAO_IP)
            elif response_text in ("rainy"):
                send_message("3", NAO_IP)
            elif response_text in ("stormy"):
                send_message("4", NAO_IP)
            elif response_text in ("cold"):
                send_message("5", NAO_IP)
            else:
                send_message("-1", NAO_IP)
                return
            
            filename = "response.wav"
            speech = make_joke(joke=joke, response=response_text, style=1)
            piper_tts(speech, filename)
            send_message_audio(filename, NAO_IP)
            
        
        elif message_arr[2] == "2":
            joke_script = "Is the {profession} a high-paying or low-paying profession. Choose between (high, low). SAY ONLY ONE WORD"
            joke = LLM_Joke(joke_script=joke_script)
            response_text = joke.main().lower()
            if response_text in ("low"):
                send_message("0", NAO_IP)
            elif response_text in ("high"):
                send_message("1", NAO_IP)
            else:
                send_message("-1", NAO_IP)
                return
            
            filename = "response.wav"
            speech = make_joke(joke=joke, response=response_text, style=2)
            piper_tts(speech, filename)
            send_message_audio(filename, NAO_IP)


        else:
            return
        
    
    if message == "end":
        print("Ending process")

        

#send_message("start", NAO_IP)

def main():
    cam_initializer()
    while True:
        print("waiting for message")
        message = receive_message()
        execute_command(message)
        if message == "end":
            print(time.time()-start)
            break
    picam0.close()
    picam1.close()
    
if __name__=="__main__":
    main()
