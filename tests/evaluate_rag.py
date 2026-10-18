import json
from pathlib import Path

from app.services.rag_service import RAGService
PDF_PATH = Path.home() / "Downloads"/ "TF_NLP_NeuralNets_StudyMaterial.pdf"
DATASET_PATH = Path("tests/evaluation_dataset.json")

def evaluate():
     if not PDF_PATH.exists():
          raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
     with DATASET_PATH.open("r", encoding="utf-8") as file:
          dataset = json.load(file)

     rag = RAGService()

     print("Indexing PDF...")
     chunk_count = rag.index_document(str(PDF_PATH))
     print(f"Indexed {chunk_count} chunks.\n")

     total_reciprocal_rank = 0.0
     total_hit = 0
     total_precision = 0.0
     evaluated = 0

     for item in dataset:
          question = item["question"]
          keywords = [
               keyword.lower()
               for keyword in item["relevant_keywords"]
          ]

          chunks = rag.retrieve(question, top_k = 5)
          
          if "TensorFlow GradientTape" in question:
            for i, chunk in enumerate(chunks, start=1):
                print(f"\nChunk {i}:\n{chunk}")

          relevant_chunks = [
               chunk
               for chunk in chunks
               if any(keyword in chunk.lower() for keyword in keywords)
          ]

          hit = int(bool(relevant_chunks))
          precision = len(relevant_chunks) / len(chunks) if chunks else 0.0

          reciprocal_rank = 0.0

          for rank, chunk in enumerate(chunks,start=1):
                if any(keyword in chunk.lower() for keyword in keywords):
                      reciprocal_rank = 1/rank
                      break

          total_reciprocal_rank += reciprocal_rank


          total_hit += hit
          total_precision += precision
          evaluated += 1

          mrr = total_reciprocal_rank / evaluated if evaluated else 0.0

          print(f"Hit@5: {hit}")
          print(f"Keyword Precision@5: {precision:.2%}")

          print(f"Question: {question}")
          print(f"Relevant chunks retrieved: {len(relevant_chunks)}/{len(chunks)}")
          print(f"Hit@5: {hit}")
          print(f"keyword Precision@5: {precision:.2%}")
          print("-" * 40)
          
     if evaluated:
          print("\n=== Evaluation SUmmary ===")
          print(f"Question evaluated: {evaluated}")
          print(f"Hit@5: {total_hit / evaluated:.2%}")
          print(f"Keyword Precision@5: {total_precision / evaluated:.2%}")
          print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
          evaluate()


