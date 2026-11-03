from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.responses import StreamingResponse
from app.ai.chat import chat_service

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Question to ask about Virat Kohli",
        example="How many international centuries has Virat Kohli scored?"
    )


@router.post("/llm")
async def chat(request: Optional[ChatRequest] = None, prompt: Optional[str] = Query(None)):
    """
    Accepts prompts via JSON body: {"prompt": "..."}
    or as query parameter: /llm?prompt=...
    Streams the exact answer directly as plain text.
    """
    user_prompt = (request.prompt if request else None) or prompt

    if not user_prompt or not user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    return StreamingResponse(
        chat_service.chat(user_prompt.strip()),
        media_type="text/plain; charset=utf-8",
        status_code=200,
    )
