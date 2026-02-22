import os
from typing import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from .agents import (
    Analyzer,
    Planner,
    Debugger,
    Coder,
    Verifier,
    Router,
    Finalyzer
)
from utility.sandbox import Sandbox, SandboxResult


class DSStarState(TypedDict):
    query: str
    workspace: str
    max_rounds: int
    max_debug_steps: int
    finalyze: bool
    guidelines: str | None
    analysis_results: dict[str, str]
    plans: list[str]
    outputs: list[str]
    base_code_filenames: list[str]
    verified: bool
    round: int
    success: bool


class DSStar:
    def __init__(self, llm: BaseChatModel, data_path: str = './data'):
        self.analyzer = Analyzer(llm)
        self.planner = Planner(llm)
        self.coder = Coder(llm)
        self.verifier = Verifier(llm)
        self.router = Router(llm)
        self.debugger = Debugger(llm)
        self.finalyzer = Finalyzer(llm)

        self.data_path = data_path
        self.data_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    def _build_graph(self):
        graph = StateGraph(DSStarState)
        graph.add_node('analyze_data', self._node_analyze_data)
        graph.add_node('init_step', self._node_init_step)
        graph.add_node('iterate', self._node_iterate)
        graph.add_node('finalize', self._node_finalize)
        graph.add_node('failed', self._node_failed)

        graph.add_edge(START, 'analyze_data')
        graph.add_edge('analyze_data', 'init_step')
        graph.add_conditional_edges(
            'init_step',
            self._route_next,
            {
                'iterate': 'iterate',
                'finalize': 'finalize',
                'failed': 'failed',
                'end': END,
            },
        )
        graph.add_conditional_edges(
            'iterate',
            self._route_next,
            {
                'iterate': 'iterate',
                'finalize': 'finalize',
                'failed': 'failed',
                'end': END,
            },
        )
        graph.add_edge('finalize', END)
        graph.add_edge('failed', END)
        return graph.compile()

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
        )

        app = self._build_graph()
        state: DSStarState = app.invoke({
            'query': query,
            'workspace': workspace,
            'max_rounds': max_rounds,
            'max_debug_steps': max_debug_steps,
            'finalyze': finalyze,
            'guidelines': guidelines,
            'analysis_results': {},
            'plans': [],
            'outputs': [],
            'base_code_filenames': [],
            'verified': False,
            'round': 0,
            'success': False,
        })

        if not state['success']:
            print("Failed to solve the task")
        else:
            print("Solved the task")
        return state['success'], state['plans'], state['base_code_filenames'][-1], state['outputs'][-1]

    def _node_analyze_data(self, state: DSStarState) -> DSStarState:
        print("Analyzing data files...")
        analysis_results: dict[str, str] = {}

        for data_file in self.data_files:
            print(f"Analyzing data file: {data_file}")
            analysis_file = self.analyzer.analyze(state['workspace'], data_file)
            result = self.sandbox.run_file(analysis_file)
            if not result.success:
                result = self._debug_analyzer(analysis_file, result.stderr, state['max_debug_steps'])

            print(f"Analysis result:\n{result.stdout}\n")
            analysis_results[data_file] = result.stdout

        print("-" * 80)
        return {**state, 'analysis_results': analysis_results}

    def _node_init_step(self, state: DSStarState) -> DSStarState:
        print("Planning initial plan...")
        initial_plan = self.planner.init_plan(state['query'], state['analysis_results'])
        print(f"Initial plan: {initial_plan}")

        print("Generating initial code...")
        initial_code_filename = self.coder.init_code(state['workspace'], state['analysis_results'], initial_plan)

        print("Running initial code...")
        result = self.sandbox.run_file(initial_code_filename)
        if not result.success:
            result = self._debug_coder(
                state['analysis_results'],
                initial_code_filename,
                result.stderr,
                state['max_debug_steps'],
            )

        plans = [initial_plan]
        outputs = [result.stdout]
        base_code_filenames = [initial_code_filename]

        print(f"Initial code results:\n{outputs[-1]}\n")
        verified = self.verifier.verify(state['query'], base_code_filenames[-1], plans, outputs[-1])
        print(f"Initial verification: {verified}\n")

        return {
            **state,
            'plans': plans,
            'outputs': outputs,
            'base_code_filenames': base_code_filenames,
            'verified': verified,
            'round': 0,
        }

    def _node_iterate(self, state: DSStarState) -> DSStarState:
        print(f"Round {state['round'] + 1} of {state['max_rounds']}")
        plans = list(state['plans'])
        outputs = list(state['outputs'])
        base_code_filenames = list(state['base_code_filenames'])

        step_index = self.router.route(state['query'], state['analysis_results'], plans, outputs[-1])
        print(f"Router answered: {f'Step {step_index}' if step_index >= 0 else 'Add step'}")
        if step_index >= 0:
            print(f"Removing {step_index} steps...")
            plans = plans[:step_index]
            outputs = outputs[:step_index]
            base_code_filenames = base_code_filenames[:step_index]

        print("Planning next step...")
        plan = self.planner.next_plan(state['query'], state['analysis_results'], plans, outputs[-1])
        print(f"Next plan: {plan}")
        plans.append(plan)

        print("Generating next code...")
        code_filename = self.coder.next_code(
            state['workspace'],
            state['analysis_results'],
            base_code_filenames[-1],
            plans,
            outputs[-1],
            state['round'],
        )
        print("Running next code...")
        result = self.sandbox.run_file(code_filename)
        if not result.success:
            result = self._debug_coder(
                state['analysis_results'],
                code_filename,
                result.stderr,
                state['max_debug_steps'],
            )
        outputs.append(result.stdout)
        print(f"Output:\n{result.stdout}\n")
        base_code_filenames.append(code_filename)

        verified = self.verifier.verify(state['query'], base_code_filenames[-1], plans, outputs[-1])
        print(f"Verification: {verified}\n")
        print("-" * 80)

        return {
            **state,
            'plans': plans,
            'outputs': outputs,
            'base_code_filenames': base_code_filenames,
            'verified': verified,
            'round': state['round'] + 1,
        }

    def _node_finalize(self, state: DSStarState) -> DSStarState:
        if not state['finalyze']:
            return {**state, 'success': True}

        final_code_filename = self.finalyzer.finalize(
            state['workspace'],
            self.data_path,
            state['query'],
            state['analysis_results'],
            state['base_code_filenames'][-1],
            state['outputs'][-1],
            state['guidelines'],
        )
        result = self.sandbox.run_file(final_code_filename)
        if not result.success:
            result = self._debug_analyzer(final_code_filename, result.stderr, state['max_debug_steps'])

        return {
            **state,
            'outputs': [*state['outputs'], result.stdout],
            'base_code_filenames': [*state['base_code_filenames'], final_code_filename],
            'success': True,
        }

    def _node_failed(self, state: DSStarState) -> DSStarState:
        return {**state, 'success': False}

    @staticmethod
    def _route_next(state: DSStarState) -> str:
        if state['verified']:
            return 'finalize' if state['finalyze'] else 'end'
        if state['round'] >= state['max_rounds']:
            return 'failed'
        return 'iterate'


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
