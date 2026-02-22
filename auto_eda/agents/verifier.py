from .agent import Agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class VerifyDecision(BaseModel):
    solved: bool = Field(description="True if the task is solved, false otherwise.")

class Verifier(Agent):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, 'verifier')
        self._parser = PydanticOutputParser(pydantic_object=VerifyDecision)

    def verify(
        self, 
        query: str, 
        code_filename: str,
        plans: list[str],
        result: str,
    ):
        with open(code_filename, 'r') as f:
            code = f.read()
        plans = '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)])
        prompt = self.prompts['verify'].format(
            question=query,
            code=code,
            plans=plans,
            result=result,
        )
        prompt = (
            f"{prompt}\n\n"
            "Return only a valid JSON object.\n"
            f"{self._parser.get_format_instructions()}"
        )
        decision = self._parser.parse(self.llm_call_raw(prompt))
        return decision.solved
