from app.services.rag_service import RAGService


def test_text_splitting():
    rag = RAGService()

    text = " ".join(
        f"word{i}" for i in range(400)
    )

    chunks = rag.split_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert "word0" in chunks[0]
    assert "word80" in chunks[1]