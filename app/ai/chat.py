import asyncio
import json
import random
from collections.abc import AsyncIterator
from app.ai.rag.pipeline import rag_pipeline
from app.ai.models import models
from app.ai.prompts.gemini_prompt import RAG_PROMPT_TEMPLATE


NO_CONTEXT_MESSAGE = "I couldn't find anything about that in the Virat Kohli document."


class ChatService:

    @staticmethod
    async def retry_with_jitter(func, *args, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0, **kwargs):
        """
        Executes an async callable with Exponential Backoff and Full Jitter.

        Why Jitter?
        When an API experiences rate limits or network issues, retrying immediately
        or at identical intervals can overwhelm the server. Adding randomness ('jitter')
        spreads out retry requests smoothly.

        Formula:
            backoff = min(max_delay, base_delay * (2 ** attempt))
            sleep_time = random.uniform(0, backoff)
        """

        for attempt in range(1, max_retries + 1):
            try:
                # Attempt to execute the function and return its result
                return await func(*args, **kwargs)

            except Exception as error:
                # If all retries are exhausted, raise the error so fallback can take over
                if attempt >= max_retries:
                    print(f"  [x] Attempt {attempt}/{max_retries} failed: {error}")
                    raise error

                # Calculate backoff with full jitter
                backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
                sleep_time = random.uniform(0, backoff)

                print(f"  [!] Attempt {attempt}/{max_retries} failed: {error}")
                print(f"      Waiting {sleep_time:.2f}s (exponential backoff + jitter) before next retry...")
                await asyncio.sleep(sleep_time)

    @staticmethod
    async def _open_stream(model_func, prompt: str) -> tuple[str, AsyncIterator[str]]:
        """
        Starts a model stream and waits for its first token, so connection/auth
        errors surface here (where they can be retried) instead of mid-answer.
        """
        stream = model_func(prompt)
        first_token = await anext(stream)
        return first_token, stream

    @staticmethod
    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    @staticmethod
    async def chat(user_prompt: str) -> AsyncIterator[str]:
        """
        Retrieves relevant chunks for the question, then streams the answer directly
        as plain text chunks from the first model in the fallback chain that responds.
        """

        # 1. Retrieve relevant chunks from the vector store
        retrieval = await rag_pipeline.retrieve(user_prompt)
        if not retrieval["results"]:
            yield NO_CONTEXT_MESSAGE
            return

        # 2. Build the RAG prompt from the retrieved context
        context, sources = rag_pipeline.build_context(retrieval["results"])
        prompt = RAG_PROMPT_TEMPLATE.format(context = context, question = user_prompt)

        print("\n" + "=" * 50)
        print("Starting LLM Request with Fallback & Retry")
        print("=" * 50)

        fallback_chain = models.get_fallback_chain()

        # For loop to traverse each model candidate in order of priority
        for model_name, model_func in fallback_chain:
            print(f"\n[Provider] Trying: {model_name}...")

            try:
                # Open the stream with our retry + jitter function (retries until the first token)
                first_token, stream = await ChatService.retry_with_jitter(
                    ChatService._open_stream, model_func, prompt
                )
            except Exception as error:
                print(f"[Fallback Triggered] {model_name} failed all retries: {error}")
                print(f"                      Switching to the next backup model in chain...")
                continue

            # Stream the answer tokens directly
            yield first_token

            try:
                async for token in stream:
                    yield token
            except Exception as error:
                # Can't switch models mid-answer, so report the error and stop
                print(f"[Error] {model_name} failed mid-stream: {error}")
                yield f"\n[Error: {model_name} failed mid-stream: {error}]"
                return

            print(f"[Status] Successfully answered by {model_name}!")
            return

        # If all models in the fallback chain fail, report it directly in the stream
        yield "All LLM models in the fallback pipeline failed. Please check credentials or network."


chat_service = ChatService()
