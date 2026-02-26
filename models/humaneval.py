from __future__ import annotations
from .benchmark import Benchmark
from docker_utils import start_docker_container, copy_code, eval_script, remove_docker_container

class HumanEval(Benchmark):
    def __init__(self):
        super().__init__()
        self._name = 'humaneval_plus'
        self._hf_name = 'evalplus/humanevalplus'
        self._hf_split = 'test'
        self._hf_revision = 'd32357cf319e50e9c8d8dab5ea876c72b0fd321b'

        self._task_id_col = 'task_id'
        self._prompt_col = 'prompt'
        self._tests_col = 'test'
        self._canonical_solution_col = 'canonical_solution'
        start_docker_container(self._container_tag, 'ganler/evalplus:v0.3.1')
    
    def __del__(self):
        remove_docker_container(self._container_tag)
        
    def run_tests(self, code: str) -> tuple:
        complete_test_script = f"""{code}\n{self.tests()}"""
        filepath = copy_code(complete_test_script, self._container_tag)
        return_code, tests_output = eval_script(self._container_tag, 'python3', filepath)
        passed = True if return_code == 0 else False
        return passed, tests_output

    def task_id(self) -> str:
        return int(self._row[self._task_id_col].split("/")[-1])
    
    def tests(self) -> str:
        return self._row[self._tests_col] + f"\ncheck({self._row['entry_point']})"
    
    def canonical_solution(self) -> str:
        return self.prompt() + self._row[self._canonical_solution_col]