import requests
import json
import time

url = "http://localhost:11434/api/chat"

def llama3(prompt):
    data = {
        "model": "llama3.2",
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
    start = time.time()
    response = requests.post(url, headers=headers, json=data)
    print(time.time() - start)
    return response.json()['message']['content']

if __name__ == "__main__":
    date = "Winter"
    location = "Portland, OR"
    request = f"Based on the likely actual weather in {location} in the {date}, say an option closest to the likely weather. ONLY SAY ONE OF THESE OPTIONS AND NOTHING MORE (Sunny, Cold, Rainy, Stormy, Overcast). SAY ONLY ONE WORD"
    response = llama3(request)
    print(request)
    print(response)

