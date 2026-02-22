from typing import Any, List, Optional
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class GatewayChatModel(BaseChatModel):
    model: str
    base_url: str
    api_key: str

    @property
    def _llm_type(self) -> str:
        return "gateway-openai-compatible"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        payload = [{"role": "user", "content": m.content} for m in messages if isinstance(m, HumanMessage)]
        resp = client.chat.completions.create(model=self.model, messages=payload)

        choices = getattr(resp, "choices", None)

        if not choices:
            raw = getattr(resp, "response", None)
            if isinstance(raw, dict):
                choices = raw.get("choices")

        if not choices:
            raise ValueError(f"No choices in provider response: {resp}")

        msg = choices[0]["message"]["content"] if isinstance(choices[0], dict) else choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=msg))])
