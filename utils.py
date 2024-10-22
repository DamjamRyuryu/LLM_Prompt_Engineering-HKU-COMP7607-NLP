import time
import itertools
from openai import OpenAI
from tqdm import tqdm
from modified_execution import check_correctness
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import copy
import numpy as np


URL = "https://api.sambanova.ai/v1"
API_KEY = "e439dfc0-6235-400f-83d7-9847afba0b57"
MODEL = "Meta-Llama-3.1-8B-Instruct"
HUMAN_EVAL = "pre_generated_data/HumanEval.jsonl"

class LlamaModel:
    def __init__(self, url, api_key, temperature: float = 1.0, attempts=1, top_p=1.0):
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model = MODEL
        self.request_interval = 1.2  # seconds
        self.retry_times = 3 # retry x times if bad connections happens.
        self.retry_interval = 3 # interval between retries
        self.temperature = temperature
        self.attempts = attempts
        self.top_p = top_p

    def request_response(self, prompt_list):
        output_list = []
        progress_bar = tqdm(prompt_list, desc='Requesting')
        for prompt in progress_bar:
            for attempt in range(self.attempts):
                for _ in range(self.retry_times):
                    try:
                        completion = self.client.chat.completions.create(
                            model= self.model,
                            messages= prompt,
                            timeout= 30,
                            temperature= self.temperature,
                            top_p=self.top_p,
                            stream= True
                        )
                        full_response = ""
                        for chunk in completion:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content'):
                                full_response += delta.content
                    except:
                        progress_bar.set_description(f'Error occurs, retry in {self.retry_interval} seconds')
                        for cnt in range(self.retry_interval-1, -1, -1):
                            time.sleep(1)
                            progress_bar.set_description(f'Error occurs, retry in {cnt} seconds')
                        progress_bar.set_description('Requesting')
                    else:
                        break
                output_list.append(full_response)
                time.sleep(self.request_interval)
        return output_list

def get_batch(in_prompt: list[dict], batch_size = 5):
    """
    keep the attempts param of the model at 1, use this function to generate batch for single problem
    which is to multiply each input by batch_size
    """
    _output = []
    for line in in_prompt:
        _extend_list = [copy.deepcopy(line) for _ in range(batch_size)]
        _output.extend(_extend_list)
    return _output

def solution_to_completion(solution: str) -> str:
    _completion = solution.strip("'''")
    # _completion = _completion[1] if len(_completion) == 3 else "'''".join(_completion[1:-1])  # obtain the completion
    # erase the function definition line (already included in the problem prompt)
    # if not _completion.startswith('def '):
    #     _completion = _completion[_completion.find('\ndef') + 1:]
    # _completion = _completion[_completion.find('\n') + 1:]
    # erase checking function (check(candidate))
    if "\ndef check(candidate):" in _completion:
        _completion = _completion[:_completion.find("\ndef check(candidate):")]
    return _completion

def concatenate_dict(in_dict_list: list[dict], append_dict_list: list[dict], new_keys: list[str], append_keys: list[str], has_indices: bool= False):
    """
    extend the original dict one by one, or with indices
    append_keys are the keys in append_dict which need to be inserted in in_dict
    new_keys are the names of the key for the newly inserted item
    please pay attention to the order
    """
    assert len(new_keys) == len(append_keys), f'Error: Wrong length of keys when concatenating. len(new) <{len(new_keys)}> should match with len(append) <{len(append_keys)}>'
    assert len(in_dict_list) <= len(append_dict_list) or has_indices, f'Error: Wrong length of input lists when concatenating without indices. len(in) <{len(in_dict_list)}> should not be larger than len(append) <{len(append_dict_list)}>'
    if has_indices:  # TODO: This is the last part of the self_evolve program, if you want to further the iteration, remember to modify this part
        for line in append_dict_list:
            assert 'index' in line, 'Error: index not found'
            i = line['index']
            for j in range(len(new_keys)):
                del in_dict_list[i]['index']
                in_dict_list[i]['prompt'] = in_dict_list[i].pop("sub_prompt_2")
                in_dict_list[i][new_keys[j]] = line[append_keys[j]]
    else:
        for i, line in enumerate(in_dict_list):
            for j in range(len(new_keys)):
                line[new_keys[j]] = append_dict_list[i][append_keys[j]]

def concatenate_str(in_dict_list: list[dict], append_str_list: list[str], key: str, processing: bool= False):
    """
    similar usage with the above function. Used when concatenating model response to the dict list
    if processing is enabled, the strings (solution) will be modified (cut the unnecessary beginning part) to completion
    """
    assert len(in_dict_list) == len(append_str_list), f'Error: Wrong length of keys when concatenating. len(new) <{len(in_dict_list)}> should match with len(append) <{len(append_str_list)}>'
    if processing:
        for i, line in enumerate(in_dict_list):
            response = solution_to_completion(append_str_list[i])
            line[key] = response
    else:
        for i, line in enumerate(in_dict_list):
            line[key] = append_str_list[i]

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def check_testcase(
    completions: list[dict],
    problems: list[dict],
    n_workers: int = 4,
    timeout: float=3.0,
    k: List[int] = [1, 3, 10],
):
    """
    construct the 1st test case of each problem and then test it with the completion of the model
    """
    # construct problem
    temp_dict = {d['task_id']: d for d in problems}
    # clean completion
    for sample in completions:
        sample["completion"] = solution_to_completion(sample["completion"])
    # Copied from evaluation.py (evaluate_functional_correctness)
    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in completions:
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (temp_dict[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        # assert len(completion_id) == len(temp_dict), "Some problems are not attempted."
        print(completion_id)

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

        # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    for sample in completions:
        task_id = sample["task_id"]
        result = results[task_id].pop(0)
        sample["result"] = result[1]["result"]
        sample["passed"] = result[1]["passed"]
        sample["test"] = temp_dict[task_id]["test"]
        sample["entry_point"] = temp_dict[task_id]["entry_point"]
    print(f'Test complete. {pass_at_k}')
    return completions

def slow_print(string :str):
    """
    sleep for 1 second after printing to prevent conflict between print and tdqm
    """
    print(string)
    time.sleep(1)
