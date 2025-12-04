from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
from openai import OpenAI

from engine_base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API Key 가 없습니다.")
        super().__init__(model=model)
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=float)

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        payload_messages = list(messages)
        if context:
            payload_messages = [
                {
                    "role": "system",
                    "content": (
                        "아래 PDF 컨텍스트를 활용하여 사용자의 질문에 답변하세요. "
                        "답변에는 참고한 내용을 자연스럽게 녹여 주세요.\n\n"
                        f"{context}"
                    ),
                }
            ] + payload_messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload_messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
