"""Inference REST endpoints (OpenAI-compatible)."""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse

from api.auth.permissions import get_current_user, Permission


router = APIRouter()


class Message(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)."""
    model: str
    messages: List[Message]
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)


class CompletionChoice(BaseModel):
    """Completion choice."""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str
    input: str | List[str]


class Embedding(BaseModel):
    """Single embedding."""
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Usage


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create chat completion (OpenAI-compatible)."""
    try:
        # TODO: Implement actual inference via TGI/vLLM
        import time
        import uuid

        # Mock response for now
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content="This is a mock response from ModelOps."
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25
            )
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create embeddings."""
    try:
        # TODO: Implement actual embedding generation
        import numpy as np

        # Convert single string to list
        inputs = [request.input] if isinstance(request.input, str) else request.input

        # Mock embeddings (768-dim)
        embeddings = [
            Embedding(
                index=i,
                embedding=np.random.randn(768).tolist()
            )
            for i in range(len(inputs))
        ]

        return EmbeddingResponse(
            data=embeddings,
            model=request.model,
            usage=Usage(
                prompt_tokens=sum(len(inp.split()) for inp in inputs),
                completion_tokens=0,
                total_tokens=sum(len(inp.split()) for inp in inputs)
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models."""
    try:
        # TODO: Implement actual model listing
        return {
            "object": "list",
            "data": [
                {
                    "id": "llama-2-7b-chat",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "modelops"
                }
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )
