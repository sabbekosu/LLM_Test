import os
import socket
import struct
import select

# Configuration
UDP_IP = "10.42.0.1"
UDP_PORT = 2033  # Change if needed
BUFFER_SIZE = 4096  # Buffer for receiving data
RECEIVED_TEXT_FILE = "received_texts.txt"
RECEIVED_FILE_DIR = "received_files"

# Ensure save directory exists
os.makedirs(RECEIVED_FILE_DIR, exist_ok=True)

# Setup UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", UDP_PORT))  # Listen on all available interfaces

def send_text(text):
    """Send a text message over UDP."""
    sock.sendto(text.encode(), (UDP_IP, UDP_PORT))
    print(f"Sent: {text}")

def receive_data():
    """Receive a WAV file over UDP and save it correctly."""
    buffer = {}  # Dictionary to store received packets
    receiving = False
    sender_addr = None  # Ensure this variable is initialized
    timeout_seconds = 30  # Timeout duration

    while True:

        ready = select.select([sock], [], [], timeout_seconds)
        if not ready[0]:  # If no data arrives within the timeout
            print("Timeout: No data received within 5 seconds.")
            return -1  # Return -1 to indicate timeout

        data, addr = sock.recvfrom(1032)  # 4-byte sequence number + 1024-byte chunk
        
        
        # Save sender's address (only once)
        if sender_addr is None:
            sender_addr = addr

        # Try decoding as text
        try:
            decoded_text = data.decode()
            if decoded_text == "EOF":
                break  # Stop receiving when EOF is detected
            else:
                print(f"Received text: {decoded_text}")  # Handle text messages
                break
        except UnicodeDecodeError:
            pass  # Binary data detected → assume it's audio data

        # Extract sequence number (first 4 bytes)
        sequence_num = struct.unpack("!I", data[:4])[0]
        chunk_data = data[4:]

        # Store packet in buffer
        buffer[sequence_num] = chunk_data
        receiving = True

    # Save received file if data was collected
    if buffer and receiving:
        sorted_data = [buffer[i] for i in sorted(buffer.keys())]
        
        # Generate unique filename
        file_index = len(os.listdir(RECEIVED_FILE_DIR))
        file_path = os.path.join(RECEIVED_FILE_DIR, f"received_audio_{file_index}.wav")

        with open(file_path, "wb") as f:
            for chunk in sorted_data:
                f.write(chunk)

        print(f"Received and saved audio file: {file_path}")
    else:
        print("Received EOF but no data buffer found.")

    # Clear buffer for next file transfer
    buffer.clear()

def main():
    print("Type 'stop' to exit.")
    while True:
        # Get user input
        message = input("Enter message to send: ").strip()
        
        if message.lower() in ("stop", "end"):
            send_text(message)
            print("Exiting...")
            break
        
        send_text(message)
        print("Waiting for response...")
        
        # Receive and process response
        receive_data()
        #print("Waiting for response...")
        #receive_data()

if __name__ == "__main__":
    main()