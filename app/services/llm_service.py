from groq import AsyncGroq

from app.config import settings


class LLMService:
    def __init__(self):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=20.0,
            max_retries=1,
        )

    async def generate_answer(self, question: str) -> str:
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Answer clearly and accurately."
                },
                {
                    "role": "user",
                    "content": question
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content or ""