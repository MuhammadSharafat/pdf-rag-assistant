from fastembed import TextEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import tempfile
import os
from pathlib import Path

load_dotenv()

# fastembed uses ONNX Runtime instead of PyTorch — no API key, no per-call
# cost, and a much smaller memory footprint (important for free-tier hosts
# like Render, which cap RAM at 512MB and can't fit a PyTorch runtime).
# Same underlying model and 384-dim output as before, so Qdrant is unaffected.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = TextEmbedding(model_name=EMBED_MODEL)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(pdf_bytes: bytes):
    # Takes raw PDF bytes rather than a local file path: in production the
    # frontend (Streamlit Cloud) and backend (this service) run on different
    # machines, so a path on one server means nothing to the other. The bytes
    # travel inside the event payload instead. PDFReader still needs an
    # actual file on disk, so we write to a short-lived temp file here.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        docs = PDFReader().load_data(file=Path(tmp_path))
        texts = [d.text for d in docs if getattr(d, "text", None)]
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))
        return chunks
    finally:
        os.unlink(tmp_path)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = list(_model.embed(texts))
    return [e.tolist() for e in embeddings]