# OCR Quality Checker with LLaMA + Vector DB Memory

import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions
import json
import uuid

# --- CONFIG ---
PDF_DIR = "./input_pdfs"
OUTPUT_DIR = "./output"
MODEL_PATH = "./models/llama-3-70b.Q4_K_M.gguf"  # Change to your quantized model
PROMPT_TEMPLATE = "./ocr_analysis_prompt.txt"

# --- SETUP ---
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,     # Use all layers on GPU
    n_batch=512,         # Increased batch size for better throughput
    n_ctx=4096,          # Context window size
    offload_kqv=True,    # Offload KQV matrices to GPU
    use_mlock=True,      # Lock memory to prevent swapping
    n_threads=8          # CPU threads for operations
)

client = chromadb.Client()
client.delete_collection("ocr_memory")
memories = client.create_collection("ocr_memory")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# --- UTILS ---
def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def analyze_text_with_llama(prompt_txt: str, ocr_text: str) -> dict:
    full_prompt = prompt_txt + ocr_text
    response = llm(full_prompt, max_tokens=512, stop=["\n\n"], echo=False)
    output = response["choices"][0]["text"].strip()
    try:
        parsed = json.loads(output)
        return parsed
    except json.JSONDecodeError:
        return {"rescan": True, "confidence": 0.0, "issues": ["Invalid JSON from model"], "summary": output}

def store_memory(ocr_text: str, analysis: dict, page_id: str):
    metadata = {
        "page_id": page_id,
        "summary": analysis.get("summary", ""),
        "rescan": analysis.get("rescan", True),
        "confidence": analysis.get("confidence", 0.0),
        "issues": analysis.get("issues", [])
    }
    memories.add(
        documents=[ocr_text],
        metadatas=[metadata],
        ids=[page_id]
    )

def query_memory(question: str):
    results = memories.query(query_texts=[question], n_results=5)
    return results

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROMPT_TEMPLATE, "r") as f:
        base_prompt = f.read()

    for pdf_name in os.listdir(PDF_DIR):
        if not pdf_name.endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        print(f"Processing: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=300)

        for idx, img in enumerate(images):
            page_id = f"{pdf_name}_page_{idx + 1}_{uuid.uuid4().hex}"
            ocr_text = ocr_image(img)
            analysis = analyze_text_with_llama(base_prompt, ocr_text)
            store_memory(ocr_text, analysis, page_id)
            print(f"Page {idx + 1}:", analysis)

    # Optional query
    print("\nQuery Example: Pages with garbled text")
    result = query_memory("pages with broken or unreadable text")
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        print("-", meta["page_id"], "->", meta["summary"])