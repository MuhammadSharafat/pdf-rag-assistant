from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import tempfile
import os
from pathlib import Path

load_dotenv()

# Runs locally on your machine — no API key, no per-call cost.
# Downloads once (~90MB) on first use, then works fully offline.
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # NOTE: was 3072 with text-embedding-3-large — your Qdrant
                 # collection's vector size must match this new value.

_model = SentenceTransformer(EMBED_MODEL)

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
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()