from __future__ import annotations

import json
import http.client
import time
from typing import List, Dict, Optional

import numpy as np
import requests

from engine_base import BaseEngine


class Embedding:
    def __init__(self, host: str, api_key: str, request_id: str) -> None:
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": self._api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request(
            "POST",
            "/v1/api-tools/embedding/v2",
            json.dumps(completion_request),
            headers,
        )
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding="utf-8"))
        conn.close()
        return result

    def execute(self, completion_request: Dict):
        res = self._send_request(completion_request)
        if res.get("status", {}).get("code") == "20000":
            return res["result"]
        return "Error"


class CompletionExecutor:
    def __init__(
        self,
        host: str,
        api_key: str,
        request_id: str,
        model_name: str,
    ) -> None:
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        self._model_name = model_name

    def execute(self, completion_request: Dict) -> Dict:
        headers = {
            "Authorization": self._api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        }

        full_content = ""

        with requests.post(
            self._host + f"/v3/chat-completions/{self._model_name}",
            headers=headers,
            json=completion_request,
            stream=True,
            timeout=60,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data:"):
                    continue

                json_part = decoded[6:].strip()
                if '"data":"[DONE]"' in json_part:
                    break

                try:
                    if not json_part.startswith("{"):
                        if json_part.startswith('"message":'):
                            json_part = "{" + json_part

                    try:
                        data = json.loads(json_part)
                    except json.JSONDecodeError as e:
                        print("JSON 파싱 실패")
                        print(f"오류 메시지: {e.msg}")
                        print(
                            f"문제 부분: →{json_part[max(0, e.pos-20):e.pos+20]}←"
                        )
                        print(f"전체 문자열: {json_part}")
                        continue

                    message = data.get("message", {})
                    content = message.get("content", "")
                    if content:
                        full_content += content
                except json.JSONDecodeError:
                    continue

        return {"result": {"message": {"content": full_content.strip()}}}


def get_hcx_embeddings(api_key: str, texts: List[str]) -> np.ndarray:
    completion_executor = Embedding(
        host="clovastudio.stream.ntruss.com",
        api_key=f"Bearer {api_key}",
        request_id="543d766ecc044afb9b3d3835e188f00b",
    )

    all_embeddings: List[List[float]] = []

    for i, text in enumerate(texts):
        if len(text) > 3000:
            text = text[:3000] + "..."

        request_data = {"text": text}

        success = False
        for retry in range(3):
            try:
                raw_result = completion_executor.execute(request_data)
                if raw_result != "Error" and "embedding" in raw_result:
                    all_embeddings.append(raw_result["embedding"])
                    success = True
                    break
                print(f"청크 {i+1} 응답 오류: {raw_result}")
            except Exception as e:
                print(f"청크 {i+1} 예외: {e}")

            if retry < 2:
                time.sleep(1)

        if not success:
            raise ValueError(f"청크 {i+1} 임베딩 완전 실패")

    return np.array(all_embeddings, dtype=np.float32)


class HCXEngine(BaseEngine):
    def __init__(
        self,
        api_key: str,
        model: str = "HCX-007",
    ) -> None:
        if not api_key:
            raise ValueError("HCX API Key 가 없습니다.")

        super().__init__(model=model)
        self.api_key = api_key
        self.model_name = model

        self.completion_executor = CompletionExecutor(
            host="https://clovastudio.stream.ntruss.com",
            api_key=f"Bearer {api_key}",
            request_id="e15290bd91554d4a96a15416d5b50c84",
            model_name=model,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        return get_hcx_embeddings(self.api_key, texts)

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> str:
        payload_messages: List[Dict[str, str]] = list(messages)

        if context:
            system_msg = {
                "role": "system",
                "content": (
                    "아래는 PDF 문서에서 추출한 컨텍스트입니다.\n"
                    "이 내용을 참고해 사용자의 질문에 답변해 주세요.\n"
                    "모르면 모른다고 답변해도 됩니다.\n\n"
                    f"{context}"
                ),
            }
            payload_messages = [system_msg] + payload_messages

        request_data = {
            "messages": payload_messages,
            "maxCompletionTokens": 1024,
            "temperature": 0.3,
            "topP": 0.9,
            "repetitionPenalty": 1.0,
            "includeAiFilters": True,
            "seed": 42,
        }

        response = self.completion_executor.execute(request_data)
        return response["result"]["message"]["content"]
