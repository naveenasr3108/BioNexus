from rag import RAG
from contradiction import ContradictionDetector
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from pdf_processor import PDFProcessor
from ml_models import PaperClassifier


app = FastAPI(title="BioNexus API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173"
],
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

print("Loading embedding model...")
embedder = EmbeddingGenerator()
print("Embedding model ready")

vector_store = VectorStore()
print("Vector store ready")

contradiction_detector = ContradictionDetector()
print("Contradiction detector ready")

rag = RAG()
print("RAG system ready")

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
    print("Starting classification...")
    classification = classifier.classify(result["full_text"])
    print("Classification complete")

# Embed (BGE-M3)
    print("Starting BGE-M3 embedding...")
    vectors = embedder.generate(chunks)
    print("BGE-M3 embedding complete")

    # Store embeddings in Qdrant
    print("Storing vectors in Qdrant...")
    stored = vector_store.add_chunks(
        file_id=file_id,
        title=result["title"],
        chunks=chunks,
        vectors=vectors
    )
    print("Qdrant storage complete")

    return {
            "file_id": file_id,
            "title": result["title"],
            "num_pages": result["num_pages"],
            "predicted_domain": f"{classification['predicted_domain']} ({int(classification['confidence']*100)}%)",
            "top3_domains": classification["top3"],
            "num_chunks": len(chunks),
            "stored_vectors": stored,
            "embedding_shape": [len(vectors), len(vectors[0])]
        }
@app.get("/search")
def search_papers(query: str, top_k: int = 5):

    query_vector = embedder.embed_query(query)

    results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k
    )

    return {
        "query": query,
        "results": results
    }
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/stats")
def stats():
    return vector_store.collection_stats()

@app.get("/ask")
def ask_question(query: str, top_k: int = 5):

    query_vector = embedder.embed_query(query)

    results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k
    )

    answer = rag.generate_answer(
        query=query,
        retrieved_chunks=results
    )

    return {
        "query": query,
        "answer": answer,
        "sources": results
    }
@app.get("/contradictions")
def detect_contradictions(query: str, top_k: int = 5):

    query_vector = embedder.embed_query(query)

    results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k
    )

    comparisons = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):

            comparison = contradiction_detector.compare(
                results[i]["text"],
                results[j]["text"]
            )

            comparisons.append({
                "source_1": results[i]["title"],
                "source_2": results[j]["title"],
                "label": comparison["label"],
                "score": comparison["score"]
            })

    return {
        "query": query,
        "comparisons": comparisons
    }