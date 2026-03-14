from __future__ import annotations
from .benchmark import Benchmark
from docker_utils import start_docker_container, copy_code, eval_script, remove_docker_container
import pandas as pd
import os

class MCEvalHard(Benchmark):
    def __init__(self, filepath="./benchmarks/mceval_hard.jsonl"):
        super().__init__()
        self._name = 'mceval_hard'
        self._filepath = filepath

        # Map to the specific keys in your JSONL file
        self._task_id_col = 'task_id' 
        self._prompt_col = 'prompt'
        self._tests_col = 'tests'               
        self._canonical_solution_col = 'canonical_solution'
        
        # Start the docker container for safe execution
        start_docker_container(self._container_tag, 'ganler/evalplus:v0.3.1')
    
    def __del__(self):
        remove_docker_container(self._container_tag)

    def load_data(self):
        """Overrides the default HF dataset loader to read your local JSONL file."""
        if not os.path.exists(self._filepath):
            raise FileNotFoundError(f"Could not find local benchmark at {self._filepath}")
        return pd.read_json(self._filepath, lines=True)
        
    def run_tests(self, code: str) -> tuple:
        # Combine the AI's code with the test suite
        complete_test_script = f"{code}\n{self.tests()}"
        filepath = copy_code(complete_test_script, self._container_tag)
        return_code, tests_output = eval_script(self._container_tag, 'python3', filepath)
        
        # If the return code is 0, the script ran without assertion errors!
        passed = True if return_code == 0 else False
        return passed, tests_output

    def task_id(self) -> str:
        # MCEval has an 'id' (0) and a 'task_id' ("C#/18"). 
        # Returning 'task_id' keeps it identifiable.
        return self._row[self._task_id_col]
    
    def tests(self) -> str:
        # Unlike HumanEval, MCEval already includes the execution block 
        # (e.g. `test_check()`) inside the test string, so we don't need 
        # to manually append anything to the end!
        return self._row[self._tests_col]
    
    def canonical_solution(self) -> str:
        return self.prompt() + self._row[self._canonical_solution_col]