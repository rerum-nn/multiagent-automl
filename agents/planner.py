from agents.agent import Agent
from openai import OpenAI

class Planner(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'planner')

    def init_plan(self, question: str, descriptions: dict[str, str]):
        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])
        plan = self.llm_call('init_plan', question=question, filenames=filenames, files_descriptions=files_descriptions)
        return plan.strip()

    def next_plan(self, question: str, descriptions: dict[str, str], plans: list[str], result: str):
        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])
        plans = '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)])
        plan = self.llm_call(
            'next_plan', 
            question=question, 
            filenames=filenames, 
            files_descriptions=files_descriptions, 
            plans=plans, 
            result=result,
        )
        return plan.strip()
        