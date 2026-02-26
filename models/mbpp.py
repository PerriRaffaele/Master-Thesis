from __future__ import annotations
from .benchmark import Benchmark
from docker_utils import start_docker_container, copy_code, eval_script, remove_docker_container

class MBPP(Benchmark):
    def __init__(self):
        super().__init__()
        self._name = 'mbpp_plus'
        self._hf_name = 'evalplus/mbppplus'
        self._hf_split = 'test'
        self._hf_revision = 'b2d74c91837c3f2a20c1299ae98133cbe7cfa077'

        self._task_id_col = 'task_id'
        self._tests_col = 'test'
        self._canonical_solution_col = 'code'
        start_docker_container(self._container_tag, 'ganler/evalplus:v0.3.1')
    
    def __del__(self):
        remove_docker_container(self._container_tag)

    def run_tests(self, code: str) -> str:
        complete_test_script = f"""{code}\n{self.tests()}"""
        filepath = copy_code(complete_test_script, self._container_tag)
        return_code, tests_output = eval_script(self._container_tag, 'python3', filepath)
        passed = True if return_code == 0 else False
        return passed, tests_output
    
    def prompt(self):
        test_lines = self.tests().strip().split("\n")
        function_name = test_lines[-1].split("assertion(")[1].split("(*inp)")[0]

        canonical_solution, signature = self._row["code"], ""
        lines_canonical = canonical_solution.split("\n")
        for line in lines_canonical:
            if function_name in line:
                signature = line
                break
        
        assert signature, f"Could not extract function signature for prompt {self._row['prompt']}"

        docstring = f'''\t"""{self._row['prompt']}"""'''
        prompt = f"{signature}\n{docstring}"
        return prompt