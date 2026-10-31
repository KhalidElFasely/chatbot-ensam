"""Step 2 - Chunk the scraped pages and build the search index.

Creates in data/index/:
  chunks.json   the text chunks with their page title and URL
  index.faiss   dense vectors (multilingual embeddings, cosine similarity)
The keyword (BM25) index is rebuilt from chunks.json at load time - it takes < 1 s.

Usage:  python build_index.py
"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_pages(path=config.PAGES_FILE):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_pages(pages, size=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP):
    """Word windows with overlap. Each chunk is prefixed with its page title, which
    gives both the embedding model and Claude the context of where the text comes from."""
    chunks = []
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            body = " ".join(words[start:start + size])
            chunks.append({
                "id": len(chunks),
                "title": p["title"],
                "url": p["url"],
                "text": f"{p['title']}\n{body}",
            })
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


def build(chunks):
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = embedder.encode([c["text"] for c in chunks], batch_size=32,
                              show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
    index.add(np.asarray(vectors, dtype="float32"))
    return index


def save(chunks, index, out_dir=config.INDEX_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": config.EMBEDDING_MODEL, "chunks": chunks}, f, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {out_dir}")


if __name__ == "__main__":
    pages = load_pages()
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    save(chunks, build(chunks))
