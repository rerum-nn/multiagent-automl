from .agent import Agent
from langchain_core.language_models.chat_models import BaseChatModel
import os
from .utils import find_code_and_delete_quotes

class Coder(Agent):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, 'coder')

    def init_code(self, workspace: str, files_descriptions: dict[str, str], plan: str):
        filenames = ' '.join(files_descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in files_descriptions.items()])
        response = self.llm_call('init_code', filenames=filenames, files_descriptions=files_descriptions, plan=plan)
        code = find_code_and_delete_quotes(response)
        if code is None:
            raise ValueError("No code block found in the response")
        
        out_filename = "init_code.py"
        os.makedirs(workspace, exist_ok=True)

        with open(os.path.join(workspace, out_filename), 'w') as f:
            f.write(code)

        return os.path.join(workspace, out_filename)

    def next_code(
        self, 
        workspace: str, 
        files_descriptions: dict[str, str],
        base_code_filename: str,  
        plans: list[str], 
        next_step: str, 
        step: int | None = None,
    ):
        filenames = ' '.join(files_descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in files_descriptions.items()])
        
        with open(base_code_filename, 'r') as f:
            base_code = f.read()

        plans = '\n'.join([f'{i+1}. {plan}' for i, plan in enumerate(plans)])
        
        response = self.llm_call(
            'next_code', 
            filenames=filenames, 
            files_descriptions=files_descriptions, 
            base_code=base_code, 
            plans=plans, 
            next_step=next_step
        )
        code = find_code_and_delete_quotes(response)
        if code is None:
            raise ValueError("No code block found in the response")

        if step is not None:
            out_filename = f"step_{step}.py"
        else:
            out_filename = "next_step.py"

        os.makedirs(workspace, exist_ok=True)

        with open(os.path.join(workspace, out_filename), 'w') as f:
            f.write(code)

        return os.path.join(workspace, out_filename)
