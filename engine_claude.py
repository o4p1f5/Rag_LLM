from __future__ import annotations

import os
from typing import List, Dict, Optional

from anthropic import Anthropic

from engine_base import BaseEngine


class ClaudeCitationsEngine(BaseEngine):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20240620",
    ) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API Key 가 없습니다.")
        super().__init__(model=model)
        self.client = Anthropic(api_key=api_key)

    @property
    def supports_embedding(self) -> bool:
        return False

    def embed(self, texts: List[str]):
        raise NotImplementedError("Claude Citations 엔진은 임베딩을 사용하지 않습니다.")

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        if not context:
            raise ValueError("ClaudeCitations 는 PDF base64 컨텍스트가 필요합니다.")

        user_question = ""
        for msg in messages:
            if msg["role"] == "user":
                user_question = msg["content"]
        if not user_question and messages:
            user_question = messages[-1]["content"]

        response = self.client.messages.create(
            model=self.model,
            system=(
                "당신은 PDF 문서의 내용을 바탕으로 사용자의 질문에 답변하는 조수입니다. "
                "가능하다면 답변에 인용 근거를 함께 제시해 주세요."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": context,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_question,
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        parts = []
        for item in response.content:
            if item.type == "text":
                parts.append(item.text)
        answer = "".join(parts)
        return answer
