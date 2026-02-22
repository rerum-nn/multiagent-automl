from .agent import Agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class RouteDecision(BaseModel):
    action: Literal['add_step', 'step'] = Field(
        description="Choose 'add_step' to append a new plan step, or 'step' to roll back to a specific step."
    )
    step_index: int | None = Field(
        default=None,
        description="Required when action='step'. 0-based step index to keep up to (exclusive truncation behavior in DSStar).",
    )


class Router(Agent):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, 'router')
        self._parser = PydanticOutputParser(pydantic_object=RouteDecision)

    def route(self, query: str, descriptions: dict[str, str], plans: list[str], result: str):
        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])
        plans = '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)])
        prompt = self.prompts['route'].format(
            question=query,
            filenames=filenames,
            files_descriptions=files_descriptions,
            plans=plans,
            result=result,
        )
        prompt = (
            f"{prompt}\n\n"
            "Return only a valid JSON object.\n"
            f"{self._parser.get_format_instructions()}"
        )
        decision = self._parser.parse(self.llm_call_raw(prompt))

        if decision.action == 'add_step':
            return -1
        if decision.step_index is None:
            raise ValueError("Router returned action='step' without step_index")
        return decision.step_index
