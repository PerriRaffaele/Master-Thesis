from __future__ import annotations
from abc import ABC, abstractmethod
from datasets import load_dataset

import os, uuid, pandas as pd

class Benchmark(ABC):
    def __init__(self):
        self._name = ''

        # HuggingFace dataset parameters
        self._hf_name = ''
        self._hf_split = ''
        self._hf_revision = ''

        # Local dataset parameters
        self._row = dict()
        self._task_id_col = ''
        self._prompt_col = ''
        self._tests_col = ''
        self._canonical_solution_col = ''
        self._container_tag = f'code-generation-container-{uuid.uuid4()}'
        
    @abstractmethod
    def run_tests(self, code: str) -> tuple:
        """
        Run the tests on the code and return the results.

        Args:
            code: The implementation of the task to test.

        Returns:
            passed, output (tuple): A tuple containing a boolean indicating if the tests passed and the output of the tests.
        """
        pass

    @property
    def row(self) -> str:
        return self._row
    
    @row.setter
    def row(self, row: pd.DataFrame):
        self._row = row

    def task_id(self) -> str:
        return self._row[self._task_id_col]

    def prompt(self) -> str:
        return self._row[self._prompt_col]

    def tests(self) -> str:
        return self._row[self._tests_col]
    
    def canonical_solution(self) -> str:
        return self._row[self._canonical_solution_col]
    
    def empty_solution(self) -> str:                
        return self.prompt()
    
    def load_data(self) -> pd.DataFrame:
        benchmarks_dir = os.path.join("benchmarks")
        benchmark_filepath = os.path.join(benchmarks_dir, f"{self._name}_dataset.jsonl")
        if os.path.exists(benchmark_filepath):
            print(f"Loading dataset from local file ...")
            return pd.read_json(benchmark_filepath, orient="records", lines=True)
        else:
            print(f"Dataset not found locally. Downloading from remote server ...")
            os.makedirs(benchmarks_dir, exist_ok=True)
            return self._load_remote_benchmark()

    def _load_remote_benchmark(self) -> pd.DataFrame: 
        print(f"Downloading {self._hf_name} dataset [{self._hf_split}] from HuggingFace ...")
        dataset = load_dataset(self._hf_name, split=self._hf_split, revision=self._hf_revision)
        df = dataset.to_pandas()
        output_dir = os.path.join("benchmarks", f"{self._name}_dataset.jsonl")
        df.to_json(output_dir, orient="records", lines=True)
        return df