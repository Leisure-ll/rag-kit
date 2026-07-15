import json
from typing import AsyncIterator, Dict, List, Optional

import httpx

from rag_kit.models import SearchHit


SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。只能基于给定的上下文回答。
如果上下文没有答案，直接说明知识库中没有找到足够依据。
回答要简洁，并在关键结论后标注来源编号，例如 [1]。"""


def build_prompt(question: str, hits: List[SearchHit]) -> str:
    context = "\n\n".join(
        f"[{index}] 来源: {hit.chunk.metadata.get('source', 'unknown')}\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    )
    return f"问题：{question}\n\n上下文：\n{context}\n\n请基于上下文回答。"


class ChatClient:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def complete(self, question: str, hits: List[SearchHit]) -> str:
        if not self.enabled:
            return fallback_answer(question, hits)

        payload = self._payload(question, hits, stream=False)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream(self, question: str, hits: List[SearchHit]) -> AsyncIterator[str]:
        if not self.enabled:
            for token in fallback_answer(question, hits).split():
                yield token + " "
            return

        payload = self._payload(question, hits, stream=True)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    def _payload(self, question: str, hits: List[SearchHit], stream: bool) -> Dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
            "temperature": 0.2,
            "stream": stream,
        }

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def fallback_answer(question: str, hits: List[SearchHit]) -> str:
    if not hits:
        return "知识库中没有找到足够依据。请先导入文档后再提问。"

    parts = ["当前未配置大模型 API Key，以下为基于检索片段的抽取式回答："]
    for index, hit in enumerate(hits[:3], start=1):
        preview = hit.chunk.text.strip().replace("\n", " ")
        if len(preview) > 260:
            preview = preview[:260] + "..."
        parts.append(f"[{index}] {preview}")
    return "\n\n".join(parts)
