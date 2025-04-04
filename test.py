from llama_cpp import Llama

# Initialize the model
llm = Llama(
    model_path="models/codellama-70b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=60  # Adjust based on GPU VRAM available
)

# Use the chat-completion API for instruct-style models
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=32,
    temperature=0.7,
    top_p=0.95
)

# Print only the model's text response
print(response["choices"][0]["message"]["content"])
