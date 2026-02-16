import os
from openai import OpenAI
from agents import (
    Analyzer,
    Planner,
    Debugger,
    Coder,
    Verifier,
    Router,
    Finalyzer
)
from sandbox import Sandbox, SandboxResult

class DSStar:
    def __init__(self, client: OpenAI, model_name: str, data_path: str = './data'):
        self.analyzer = Analyzer(client, model_name)
        self.planner = Planner(client, model_name)
        self.coder = Coder(client, model_name)
        self.verifier = Verifier(client, model_name)
        self.router = Router(client, model_name)
        self.debugger = Debugger(client, model_name)
        self.finalyzer = Finalyzer(client, model_name)

        self.data_path = data_path
        self.data_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    
    def run(
        self, 
        query: str, 
        workspace: str = './workspace', 
        max_rounds: int = 20, 
        max_debug_steps: int = 5,
        finalyze: bool = True,
        guidelines: str | None = None
    ):
        print(f"Running DS-Star for query: {query}")
        print(f"Workspace: {workspace}")
        print(f"Data path: {self.data_path}")
        print(f"Max rounds: {max_rounds}")
        print(f"Max debug steps: {max_debug_steps}")
        print(f"Guidelines: {guidelines}")
        print("-" * 80)

        print("Initializing sandbox...")
        self.sandbox = Sandbox(
            allowed_dirs=[os.path.abspath(workspace), os.path.abspath(self.data_path)],
            timeout=60,
            project_root='.',
        )

        print("Analyzing data files...")
        analysis_results = {}
        
        for data_file in self.data_files:
            print(f"Analyzing data file: {data_file}")
            analysis_file = self.analyzer.analyze(workspace, data_file)
            result = self.sandbox.run_file(analysis_file)
            if not result.success:
                result = self._debug_analyzer(analysis_file, result.stderr, max_debug_steps)

            print(f"Analysis result:\n{result.stdout}\n")

            analysis_results[data_file] = result.stdout

        print("-" * 80)
        print("Planning initial plan...")
        initial_plan = self.planner.init_plan(query, analysis_results)
        print(f"Initial plan: {initial_plan}")

        print("Generating initial code...")
        initial_code_filename = self.coder.init_code(workspace, analysis_results, initial_plan)

        print("Running initial code...")
        result = self.sandbox.run_file(initial_code_filename)
        if not result.success:
            result = self._debug_coder(analysis_results, initial_code_filename, result.stderr, max_debug_steps)
        
        plans = [initial_plan]
        outputs = [result.stdout]
        base_code_filenames = [initial_code_filename]

        print(f"Initial code results:\n{outputs[-1]}\n")

        verified = self.verifier.verify(query, base_code_filenames[-1], plans, outputs[-1])
        print(f"Initial verification: {verified}\n")

        round = 0
        while not verified and round < max_rounds:
            print(f"Round {round + 1} of {max_rounds}")
            step_index = self.router.route(query, analysis_results, plans, outputs[-1])
            print(f"Router answered: {f'Step {step_index}' if step_index >= 0 else 'Add step'}")
            if step_index >= 0:
                print(f"Removing {step_index} steps...")
                plans = plans[:step_index]
                outputs = outputs[:step_index]
                base_code_filenames = base_code_filenames[:step_index]

            print(f"Planning next step...")
            plan = self.planner.next_plan(query, analysis_results, plans, outputs[-1])
            print(f"Next plan: {plan}")
            plans.append(plan)
            print(f"Generating next code...")
            code_filename = self.coder.next_code(workspace, analysis_results, base_code_filenames[-1], plans, outputs[-1])
            print(f"Running next code...")
            result = self.sandbox.run_file(code_filename)
            if not result.success:
                result = self._debug_coder(analysis_results, code_filename, result.stderr, max_debug_steps)
            outputs.append(result.stdout)
            print(f"Output:\n{result.stdout}\n")
            base_code_filenames.append(code_filename)
            verified = self.verifier.verify(query, base_code_filenames[-1], plans, outputs[-1])
            print(f"Verification: {verified}\n")

            round += 1

            print("-" * 80)

        if not verified:
            print("Failed to solve the task")
            return False, plans, base_code_filenames[-1], outputs[-1]

        if finalyze:
            final_code_filename = self.finalyzer.finalize(workspace, self.data_path, query, analysis_results, base_code_filenames[-1], outputs[-1], guidelines)
            result = self.sandbox.run_file(final_code_filename)
            if not result.success:
                result = self._debug_analyzer(final_code_filename, result.stderr, max_debug_steps)
            outputs.append(result.stdout)
            base_code_filenames.append(final_code_filename)

        print("Solved the task")
        return True, plans, base_code_filenames[-1], outputs[-1]


    def _debug_analyzer(self, code_filename: str, stderr: str, max_debug_steps: int):
        for _ in range(max_debug_steps):
            analysis_file = self.debugger.debug_analyzer(self.data_path, code_filename, stderr)
            result = self.sandbox.run_file(analysis_file)
            if result.success:
                return result
        else:
            raise ValueError("Failed to debug the analysis file")


    def _debug_coder(self, descriptions: dict[str, str], code_filename: str, stderr: str, max_debug_steps: int):
        for _ in range(max_debug_steps):
            coder_file = self.debugger.debug_coder(self.data_path, descriptions, code_filename, stderr)
            result = self.sandbox.run_file(coder_file)
            if result.success:
                return result
        else:
            raise ValueError("Failed to debug the coder file")
