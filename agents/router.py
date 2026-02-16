from agents.agent import Agent
from openai import OpenAI

class Router(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'router')

    def route(self, query: str, descriptions: dict[str, str], plans: list[str], result: str):
        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])
        plans = '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)])
        response = self.llm_call(
            'route', 
            question=query, 
            filenames=filenames, 
            files_descriptions=files_descriptions, 
            plans=plans, 
            result=result,
        )

        if response.strip().lower() == 'add step':
            return -1
        elif response.strip().lower().startswith('step'):
            return int(response.strip().split(' ')[1])
        else:
            raise ValueError(f"Invalid response: {response.strip()}")
