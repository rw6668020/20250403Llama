# chat_with_documents.py
# Chat with your OCR'd and analyzed PDF content using local LLaMA + Chroma (GPU-enabled)

import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama
import readline  # for command history in CLI

# --- CONFIG ---
MODEL_PATH = "./models/codellama-70b-instruct.Q4_K_M.gguf"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ocr_memory"

# --- INIT ---
print("\n🧠 Loading vector memory...")
client = chromadb.Client()
collection = client.get_or_create_collection(COLLECTION_NAME)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

print("🤖 Loading LLaMA model with GPU support (this may take a few seconds)...")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, use_mlock=True, n_threads=8)

# --- CHAT LOOP ---
def retrieve_relevant_chunks(query: str, n=4):
    results = collection.query(query_texts=[query], n_results=n)
    return results

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return f"""
You are an AI assistant helping a user understand the contents of scanned and OCR'd documents.
You have access to extracted page text and summaries from prior analysis.

Answer the user's question using the context provided.
Be concise, helpful, and avoid hallucinating information not in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def ask_llama(full_prompt):
    response = llm(full_prompt, max_tokens=512, stop=["\n\n"], echo=False)
    return response["choices"][0]["text"].strip()

# --- MAIN LOOP ---
print("\n💬 You can now ask questions about your scanned documents.")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    print("\n🔍 Retrieving relevant context...")
    retrieved = retrieve_relevant_chunks(user_input, n=4)
    docs = retrieved["documents"][0]

    if not docs:
        print("⚠️  No relevant content found. Try rephrasing your question.")
        continue

    full_prompt = build_prompt(docs, user_input)
    print("🤔 Thinking...")
    reply = ask_llama(full_prompt)

    print("\n🤖 LLaMA:", reply)