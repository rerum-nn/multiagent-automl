import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class Agent:
    def __init__(self, llm: BaseChatModel, name: str):
        self.llm = llm
        self.max_retries = int(os.environ.get('LLM_MAX_RETRIES', '2'))
        self.name = name
        self.prompt_dir = os.environ.get('PROMPT_DIR', './auto_eda/agents/prompts')
        self.prompts = self.load_prompts()

    def load_prompts(self):
        prompts = {}

        path = os.path.join(self.prompt_dir, self.name)
        for file in os.listdir(path):
            if file.endswith('.md'):
                with open(os.path.join(path, file), 'r') as f:
                    prompts[file.split('.')[0]] = f.read()

        return prompts

    def llm_call(self, prompt_name: str, *args, **kwargs):
        prompt = self.prompts[prompt_name].format(*args, **kwargs)
        return self.llm_call_raw(prompt)

    def llm_call_raw(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return self._extract_text(response.content)
            except Exception as exc:  # pragma: no cover - provider/runtime specific
                last_error = exc
                if attempt == self.max_retries:
                    break
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts") from last_error

    @staticmethod
    def _extract_text(content: str | list) -> str:
        if isinstance(content, str):
            return content

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                # Some providers return structured blocks with {"type": "text", "text": "..."}.
                if block.get('type') == 'text':
                    text_parts.append(str(block.get('text', '')))
                elif 'text' in block:
                    text_parts.append(str(block['text']))
            else:
                text_parts.append(str(block))
        return ''.join(text_parts)
