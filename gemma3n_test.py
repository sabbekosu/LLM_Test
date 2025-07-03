from llama_cpp import Llama

model_path = "gemma-3n-E2B-it-GGUF/gemma-3n-E2B-it-Q8_0.gguf"
llm = Llama(model_path=model_path, verbose=False)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "Respond with only one word of the options (Sunny, Cold, Rainy, Stormy, Overcast)"}]
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "location2.wav"},
            {"type": "text", "text": "What is ther likely weather in the audio's location in the summer?"}
        ]
    }
]
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=100,
    temperature=0.7,
)
generated_response = response['choices'][0]['message']['content']
print(generated_response)