from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, field_validator
import os
import tempfile
import logging
from groq import RateLimitError
import time

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production AI Assistant",
    version="1.0.0"
)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

logger = logging.getLogger(__name__)


from app.services.rag_service import RAGService
from app.services.llm_service import LLMService



rag_service = RAGService()
llm_service = LLMService()

answer_cache = {}
#class QuestionRequest(BaseModel):
   # question: str = Field(min_length=3, max_length=2000)

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Question to ask about the uploaded PDF"
    )

    @field_validator("question")
    @classmethod

    def validate_question(cls,value:str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

class AnswerResponse(BaseModel):
    answer: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail="Only PDF files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail= "The PDF is empty.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB."
        )
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        chunk_count = rag_service.index_document(temp_path)

        return {
            "message": "Document indexed successfully",
            "filename": os.path.basename(file.filename),
            "chunks_created": chunk_count
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask-document")
@limiter.limit("10/minute")
async def ask_document(
    request: Request,
    question_request: QuestionRequest
):

    question = question_request.question.strip()
    cache_key = question.lower()

    if cache_key in answer_cache:
        return {
            "question": question,
            "answer": answer_cache[cache_key],
            "cache": True
        }
    try:
        retrieval_start = time.perf_counter()

        chunks = rag_service.retrieve(question_request.question, top_k=3)

        retrieval_duration = time.perf_counter() - retrieval_start

        logger.info(
            "Retrieval completed | duration_seconds=%.4f | chunks=%d",
            retrieval_duration,
            len(chunks)
        )
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documnet content found."
            )

        context = "\n\n".join(chunks)

        logger.info("Received document question")

        prompt = (
            "Answer th euser's question using only the provided context."
            "i teh context does not contain the answer, say you cannot"
            "find the answer in the uploaded document.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question_request.question}"
        )

        start_time = time.perf_counter()
        answer = await llm_service.generate_answer(prompt)
        llm_duration = time.perf_counter() - start_time

        logger.info(
            "LLM generation completed ? duration_seconds=%.2f",
            llm_duration
        )

        #store the generated answer in the cache
        answer_cache[cache_key] = answer

        question = question_request.question.strip()
        cache_key = question.lower()

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": len(chunks),
            "cache": False
            }
    except HTTPException:
        raise

    except RateLimitError:
        logger.warning("Groq rate limit exceeded")

        raise HTTPException(
            status_code=429,
            detail="The AI service is temporarily rate-limited. Please try again shortly."
        )

    except Exception:
            logger.exception("Failed to process document question")

    raise HTTPException(
        status_code=502,
        detail="The AI service could not process your request. Please try again."
    )
            
    





