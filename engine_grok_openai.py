#(OpenAI 임베딩 + Grok 챗)
from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
from openai import OpenAI

from engine_base import BaseEngine


class GrokOpenAIEmbeddingEngine(BaseEngine):
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-non-reasoning",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        xai_api_key = xai_api_key or os.getenv("XAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API Key 가 없습니다.")
        if not xai_api_key:
            raise ValueError("xAI API Key 가 없습니다.")
        super().__init__(model=model)
        self.embedding_client = OpenAI(api_key=openai_api_key)
        self.grok_client = OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1",
        )
        self.embedding_model = embedding_model

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in resp.data]
        return np.array(vectors, dtype=float)

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        payload_messages = []
        if context:
            payload_messages.append(
                {
                    "role": "system",
                    "content": (
                        "아래 PDF 컨텍스트를 참고하여 질문에 답변해 주세요.\n\n"
                        f"{context}"
                    ),
                }
            )
        payload_messages.extend(messages)

        resp = self.grok_client.chat.completions.create(
            model=self.model,
            messages=payload_messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""
