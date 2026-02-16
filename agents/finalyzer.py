from .agent import Agent
from agents.utils import find_code_and_delete_quotes
from openai import OpenAI
import os

class Finalyzer(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'finalyzer')

    def finalize(
        self, 
        workspace: str,
        data_directory: str, 
        query: str, 
        descriptions: dict[str, str], 
        code_filename: str, 
        result: str, 
        guidelines: str | None = None,
    ):
        with open(code_filename, 'r') as f:
            code = f.read()
        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])
        guidelines = guidelines if guidelines is not None else ''
        response = self.llm_call(
            'finalyze', 
            data_directory=data_directory, 
            question=query, 
            filenames=filenames, 
            files_descriptions=files_descriptions, 
            code=code,
            result=result, 
            guidelines=guidelines,
        )

        final_code = find_code_and_delete_quotes(response)
        if final_code is None:
            raise ValueError("No code block found in the response")

        with open(os.path.join(workspace, 'final_code.py'), 'w') as f:
            f.write(final_code)

        return os.path.join(workspace, 'final_code.py')
