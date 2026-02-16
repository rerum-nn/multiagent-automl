from agents.agent import Agent
from openai import OpenAI

class Verifier(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'verifier')

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
        response = self.llm_call(
            'verify', 
            question=query, 
            code=code, 
            plans=plans, 
            result=result,
        )
        
        if response.strip().lower() == 'yes':
            return True
        else:
            return False
