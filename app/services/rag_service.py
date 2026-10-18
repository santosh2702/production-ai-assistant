from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "Sentence-transformers/all-MiniLM-l6-v2"
        )
        self.index = None
        self.chunks = []



    def extract_text(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        text = "\n".join(pages).strip()

        if not text:
            raise ValueError(
                "No extractable text found in the PDF."
            )

        return text

    def split_text(
            self,
            text:str,
            chunk_size: int = 150,
            overlap: int = 30,) -> list[str]:

        words = text.split()
        if overlap >= chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

        chunks = []

        for start in range(
            0, len(words), chunk_size - overlap,
        ):
            chunk = words[start:start + chunk_size]

            if chunk:
                chunks.append(" ".join(chunk))

        return chunks

    def index_document(self, file_path: str) -> int:
        text = self.extract_text(file_path)

        print("Extracted characters:", len(text))
        print("Extracted words:", len(text.split()))
        print("Generated chunks:", len(self.split_text(text)))

        self.chunks = self.split_text(text)

        embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy = True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )
        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )
        self.index.add(embeddings)
        return len(self.chunks)
    
    

    def retrieve(
            self,
            question: str,
            top_k: int = 3) -> list[str]:
        
        if self.index is None:
            raise ValueError(
                 "No documents have been indexed."
        )
        query_embedding = self.embedding_model.encode([question],
                                                      convert_to_numpy = True,
                                                      normalize_embeddings = True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(
            query_embedding, k)
        print("Similarity scores:", scores[0])
        return[
            self.chunks[i]
            for i in indices[0]
            if i >= 0
        ]


  #  User question
 #    ↓
#Embedding generate hoti hai
 #    ↓
#FAISS IndexFlatIP se compare
 #    ↓
#Top K chunks select hote hain
#     ↓
#Chunks return hote hain
        