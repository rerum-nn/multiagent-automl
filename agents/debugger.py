from agents.agent import Agent
from openai import OpenAI
from agents.utils import find_code_and_delete_quotes

class Debugger(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        super().__init__(client, model_name, 'debugger')

    def debug_analyzer(self, data_directory: str, code_filename: str, stderr: str):
        bug_summary = self.llm_call('bug_summary', bug=stderr, filename=code_filename, data_directory=data_directory)

        with open(code_filename, 'r') as f:
            code = f.read()

        response = self.llm_call('debug_analyzer', bug=bug_summary, code=code, data_directory=data_directory)
        code = find_code_and_delete_quotes(response)
        if code is None:
            raise ValueError("No code block found in the response")

        with open(code_filename, 'w') as f:
            f.write(code)

        return code_filename

    def debug_coder(self, data_directory: str, descriptions: dict[str, str], code_filename: str, stderr: str):
        bug_summary = self.llm_call('bug_summary', bug=stderr, filename=code_filename, data_directory=data_directory)

        with open(code_filename, 'r') as f:
            code = f.read()

        filenames = ' '.join(descriptions.keys())
        files_descriptions = '\n'.join([f'{file}\n{description}\n' for file, description in descriptions.items()])

        response = self.llm_call(
            'debug_coder', 
            bug=bug_summary, 
            code=code, 
            filenames=filenames, 
            files_descriptions=files_descriptions, 
            data_directory=data_directory,
        )
        code = find_code_and_delete_quotes(response)
        if code is None:
            raise ValueError("No code block found in the response")
