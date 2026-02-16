from openai import OpenAI
import os

class Agent:
    def __init__(self, client: OpenAI, model_name: str, name: str):
        self.client = client
        self.model_name = model_name
        self.name = name
        self.prompt_dir = os.environ.get('PROMPT_DIR', './agents/prompts')
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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
