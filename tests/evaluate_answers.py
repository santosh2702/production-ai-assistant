import asyncio
import json
import sys
from pathlib import Path

# Allow imports from the app directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

DATASET_PATH = Path(__file__).resolve().parent / "evaluation_dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "answer_evaluation_results.json"


async def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    rag_service = RAGService()
    llm_service = LLMService()

# Index the PDF before running evaluation
    pdf_path = Path.home() / "Downloads" / "TF_NLP_NeuralNets_StudyMaterial.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chunk_count = rag_service.index_document(str(pdf_path))

    print(f"PDF indexed successfully. Total chunks: {chunk_count}")

    results = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]

        print(f"\n[{index}/{len(dataset)}] {question}")

        try:
            chunks = rag_service.retrieve(question, top_k=3)

            if not chunks:
                print("No relevant chunks retrieved.")
                answer = ""
            else:
                context = "\n\n".join(chunks)
                

                prompt = f"""
Answer the question using only the context below.
If the context does not contain the answer, say you do not have
enough information. Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

                answer = await llm_service.generate_answer(prompt)

            result = {
                "question": question,
                "expected_answer": item.get("expected_answer", ""),
                "generated_answer": answer,
                "retrieved_chunks": len(chunks),
            }

            results.append(result)
            print(f"Answer: {answer}")

        except Exception as error:
            print(f"Error: {error}")

            results.append({
                "question": question,
                "expected_answer": item.get("expected_answer", ""),
                "generated_answer": "",
                "error": str(error),
            })

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nEvaluation completed.")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())