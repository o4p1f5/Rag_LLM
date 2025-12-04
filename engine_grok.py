# (GrokDirect / 프롬프트 스터핑 전용)

from __future__ import annotations

import os
from typing import List, Dict, Optional

from openai import OpenAI

from engine_base import BaseEngine


class GrokDirectEngine(BaseEngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-non-reasoning",
    ) -> None:
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("xAI API Key 가 없습니다.")
        super().__init__(model=model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    @property
    def supports_embedding(self) -> bool:
        return False

    def embed(self, texts: List[str]):
        raise NotImplementedError("GrokDirect 엔진은 임베딩을 사용하지 않습니다.")

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
                        "아래는 PDF 전체에서 가져온 컨텍스트입니다. "
                        "이 내용을 최대한 활용하여 사용자의 질문에 답변해 주세요.\n\n"
                        f"{context}"
                    ),
                }
            )
        payload_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload_messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
