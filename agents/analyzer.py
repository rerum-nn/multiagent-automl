from agents.agent import Agent
from openai import OpenAI

import os

from agents.utils import find_code_and_delete_quotes

class Analyzer(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'analyzer')

    def analyze(self, workspace: str, filename: str):
        response = self.llm_call('analyze', filename=filename)
        code = find_code_and_delete_quotes(response)
        if code is None:
            raise ValueError("No code block found in the response")

        filename = os.path.basename(filename)
        
        out_filename = '_'.join(filename.split('.')) + '.py'
        os.makedirs(workspace, exist_ok=True)

        with open(os.path.join(workspace, out_filename), 'w') as f:
            f.write(code)
            
        return os.path.join(workspace, out_filename)
