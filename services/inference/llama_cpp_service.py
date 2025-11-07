"""Lightweight inference service using llama.cpp for CPU inference."""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed")


class LlamaCppInferenceService:
    """Lightweight inference service using llama.cpp (CPU-optimized)."""

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """Initialize llama.cpp inference service.

        Args:
            model_path: Path to GGUF model file
            config: Optional configuration
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )

        self.model_path = model_path
        self.config = config or {}
        self.llm = None

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load GGUF model with llama.cpp."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading model from {self.model_path}...")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config.get("n_ctx", 2048),
            n_threads=self.config.get("n_threads", 4),
            n_gpu_layers=self.config.get("n_gpu_layers", 0),
            verbose=self.config.get("verbose", False)
        )

        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences

        Returns:
            Generated text and metadata
        """
        if not self.llm:
            raise RuntimeError("Model not loaded")

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False
        )

        return {
            "text": response["choices"][0]["text"],
            "tokens_generated": response["usage"]["completion_tokens"],
            "tokens_prompt": response["usage"]["prompt_tokens"],
            "model": self.model_path
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Response message and metadata
        """
        if not self.llm:
            raise RuntimeError("Model not loaded")

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return {
            "message": response["choices"][0]["message"]["content"],
            "tokens_generated": response["usage"]["completion_tokens"],
            "tokens_prompt": response["usage"]["prompt_tokens"],
            "model": self.model_path
        }
