from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from pdf_processor import PDFProcessor
from ml_models import PaperClassifier
# from embeddings import EmbeddingGenerator

app = FastAPI(title="BioNexus API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Creating PDF Processor...")
processor = PDFProcessor()
print("PDF Processor created")

print("Creating Classifier...")
classifier = PaperClassifier()
print("Classifier created")
# embedder = EmbeddingGenerator()

@app.post("/analyze")
async def analyze_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract + clean
    result = processor.extract_text_and_metadata(save_path)

    # Chunk
    chunks = processor.chunk_text(result["full_text"])

    # Classify (SciBERT)
    classification = classifier.classify(result["full_text"])

    # Embed (BGE-M3)
    # vectors = embedder.generate(chunks)

    return {
        "file_id": file_id,
        "title": result["title"],
        "num_pages": result["num_pages"],
        "predicted_domain": f"{classification['predicted_domain']} ({int(classification['confidence']*100)}%)",
        "top3_domains": classification["top3"],
        "num_chunks": len(chunks),
        # "embedding_shape": [len(vectors), len(vectors[0])]
    }

@app.get("/health")
def health():
    return {"status": "ok"}