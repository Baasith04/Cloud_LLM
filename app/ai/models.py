from collections.abc import AsyncIterator
from google import genai
from groq import AsyncGroq
from app.core.config import settings
from app.ai.prompts.gemini_prompt import SYSTEM_PROMPT


class Models:
    """
    A unified class to manage multiple LLM providers (Gemini, Groq, etc.).
    Each provider streams its response as text pieces so the server can forward them as they arrive.
    """

    def __init__(self):
        # Initialize clients using API keys from configuration
        self.gemini_client = genai.Client(api_key = settings.GEMINI_API_KEY)
        self.groq_client = AsyncGroq(api_key = settings.GROQ_API_KEY)

    async def stream_gemini(self, prompt: str, model_name: str = "gemini-3.5-flash") -> AsyncIterator[str]:
        """
        Streams Google Gemini's response, yielding text pieces as they arrive.
        Any API errors are raised so that retry and fallback logic can handle them.
        """
        response = await self.gemini_client.aio.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def stream_groq(self, prompt: str, model_name: str = "openai/gpt-oss-20b") -> AsyncIterator[str]:
        """
        Streams Groq's response, yielding text pieces as they arrive.
        Any API errors are raised so that retry and fallback logic can handle them.
        """
        chat_completion = await self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream = True,
            temperature = 1.0,
        )

        async for chunk in chat_completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def get_fallback_chain(self):
        """
        Returns an ordered list of models for the fallback pipeline.
        Format: [ (display_name, stream_function), ... ]
        """
        return [
            ("Gemini (Google)", self.stream_gemini),
            ("Groq (OpenAI/OSS)", self.stream_groq),
        ]


models = Models()
