import pdb
from typing import Iterable, Dict, List, Any
from utils import *
import gzip
import json
import os


TEMPERATURE = 0.8
TOP_P = 0.8
OUTPUTFILE = f"zeroshot_baseline_08&08_2.jsonl"
ATTEMPTS = 5

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)



def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def get_input_list(evalset_file: str = HUMAN_EVAL) -> list[dict[str, Any]]:
    problems = read_problems(evalset_file)
    prompts = []
    for p in problems:
        prompt = problems[p]["prompt"]
        task_id = problems[p]["task_id"]
        prompts.append({'prompt':prompt, 'task_id':task_id})
    return prompts

def get_prompt_list(input_list: list[dict[str, Any]]):
    prompt_lists = []
    for input_problem in input_list:
        prompt = [
            {"role": "system", "content": "Give the code implementation of user's problem. Environment: ipython"},
            {"role": "user", "content": input_problem['prompt']}
        ]
        prompt_lists.append(prompt)
    return prompt_lists

if __name__ == '__main__':
    service = LlamaModel(URL,API_KEY, TEMPERATURE, ATTEMPTS,TOP_P)
    inputs = get_input_list(HUMAN_EVAL)
    prompts = get_prompt_list(inputs)
    outputs = service.request_response(prompts)

    # construct jsonl
    assert len(inputs) == len(prompts) and len(prompts) * ATTEMPTS == len(outputs), 'Warning: wrong length'
    dicts = []
    for idx in range(len(inputs)):
        single_problem = []
        for j in range(ATTEMPTS):
            single_dict = {
                'input': inputs[idx]['prompt'],
                'prompt': prompts[idx],
                'output': solution_to_completion(outputs[idx*ATTEMPTS + j]),
                'task_id': inputs[idx]['task_id']
            }
            single_problem.append(single_dict)
        dicts.extend(single_problem)
    write_jsonl(OUTPUTFILE, dicts)
    print('DONE')