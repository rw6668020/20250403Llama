import os
from llama_cpp import Llama
import time

# Force CUDA and set debugging
os.environ["LLAMA_CUBLAS"] = "1"
os.environ["LLAMA_VERBOSE"] = "1"  # Enable verbose output

print("System environment check:")
print(f"CUDA enabled: {os.environ.get('LLAMA_CUBLAS', 'Not set')}")

# Record start time
start_time = time.time()
print("\nInitializing model with dual RTX 3090 support...")

# Initialize the model with explicit GPU settings
llm = Llama(
    model_path="models/codellama-70b-instruct.Q4_K_M.gguf",
    n_gpu_layers=80,       # Set explicit number of layers (try with exact model layers)
    n_batch=512,           # Increased batch size
    n_ctx=2048,            # Reduced context to save memory
    offload_kqv=True,      # Offload KQV matrices to GPU
    use_mlock=False,       # Don't lock memory which can cause issues
    n_threads=4,           # Reduced CPU threads
    seed=42,               # Set a seed for reproducibility
    verbose=True           # Enable verbose output
)

# Record model loading time
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Check if GPU is being used
print("\nChecking device allocation:")
if "gpu_layers" in dir(llm):
    print(f"GPU layers: {llm.gpu_layers}")
else:
    print("GPU layers attribute not found in model instance")

# Use the chat-completion API for instruct-style models
print("\nRunning inference test...")
inference_start = time.time()

response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Write a very brief hello world program in Python."}
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95
)

# Calculate inference time
inference_time = time.time() - inference_start
print(f"Inference completed in {inference_time:.2f} seconds")

# Print only the model's text response
print("\nModel response:")
print(response["choices"][0]["message"]["content"])
