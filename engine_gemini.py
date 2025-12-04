from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
from google import genai

from engine_base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        embedding_model: str = "text-embedding-004",
    ) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API Key 가 없습니다.")
        super().__init__(model=model)
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = embedding_model

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            try:
                resp = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )
                vectors.append(resp.embeddings.values)
            except Exception as exc:
                print(f"[Gemini EMBED ERROR] error={exc}")
                vectors.append([0.0])

        return np.array(vectors, dtype=float)

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        user_parts = []
        if context:
            user_parts.append(
                "아래는 PDF 문서에서 추출한 컨텍스트입니다.\n"
                "이 내용을 참고하여 뒤에 나올 질문에 답변해 주세요.\n\n"
                f"{context}\n\n"
            )

        for msg in messages:
            prefix = "사용자: " if msg["role"] == "user" else "도우미: "
            user_parts.append(prefix + msg["content"] + "\n\n")

        full_prompt = "".join(user_parts)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        if not resp.candidates:
            return ""
        return resp.candidates[0].content.parts[0].text
