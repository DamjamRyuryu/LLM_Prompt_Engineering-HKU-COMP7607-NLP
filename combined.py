from utils import *
from baseline import get_input_list, write_jsonl, stream_jsonl
from codeT import TEST_OUTPUT, match_solution_testcases, uni_agreement
from self_evolve import get_prompt_list_init, next_step_prompts
import os
import sys

OUTPUTFILE = "method_Combined.jsonl"
SYSTEM_PROMPT='''The user will ask you about Python code problem, follow their instructions. Pay attention to their required output format. Environment: ipython.'''
# following prompts templates are based on the method from this paper:https://arxiv.org/pdf/2306.02907
FIRST_STEP={
    'knowledge': "For the above question, could you briefly teach me how to solve it step by step in natural language?\nDon't write the code in this step.",
    'solution': "Based on the above idea, help me complete the function.\nBe attention, you should only output the codes without any explanation, comment, natural language and testcode.\n\
Warp your code with \"'''\""
}
SELF_REFINEMENT={
    "syntax":"When I run this code, I meet %.\nHelp me refine the code.\nYou should only output the codes without any explanation, comment, natural language and testcode.\nWrap your code with \"'''\"",
    "error":"I failed when going through the assertation:\n%\nHelp me refine the code.\nYou should only output the codes without any explanation, comment, natural language and testcode.\nWrap your code with \"'''\""
}

"""
The process is very time-consuming, create checkpoints at some stages, so you can continue next time without go through previous sections.
The checkpoints are:
1. After finishing the knowledge part (reasoning), save the responses from LLM
2. After finishing the solution part (output 1), save the responses from LLM
3. After finishing the cross verification of the solutions, save the verification results
(4) Then the code will select top k solution for each problem, concatenate all its failed testcases and request LLM for the last time,
    which is saved as final output. After that the code finish and you should run 'my_evaluation.py' to get the final pass@k
"""
CHECKPOINTS = {
    'step 1': 'combined_1_temp.jsonl',
    'step 2': 'combined_2_temp.jsonl',
    'step 3': 'combined_3_temp.jsonl',
}

if __name__ == '__main__':
    service = LlamaModel(URL, API_KEY, 0.8, 1, 0.8)
    inputs = get_input_list(HUMAN_EVAL)
    prompts = get_prompt_list_init(inputs)
    history = [{'task_id': item['task_id']} for item in inputs]
    concatenate_dict(history, inputs, ['input'], ['prompt'])
    concatenate_dict(history, prompts, ['sub_prompt_0'], ['prompt'])

    # Step 1: request for knowledge
    slow_print('first step start, ask for knowledge...')
    history = get_batch(history, 5)
    step = 0
    if not os.path.exists(CHECKPOINTS['step 1']):
        res = service.request_response([line[f'sub_prompt_{step}'] for line in history])
        write_jsonl(CHECKPOINTS['step 1'], [{'response': item} for item in res])
    else:
        res = [item['response'] for item in stream_jsonl(CHECKPOINTS['step 1'])]

    # Step 2: request for solution
    slow_print('knowledge get, ask for solution...')
    step = next_step_prompts(history, res, step)
    if not os.path.exists(CHECKPOINTS['step 2']):
        res = service.request_response([line[f'sub_prompt_{step}'] for line in history])  # the response is solution now
        write_jsonl(CHECKPOINTS['step 2'], [{'response': item} for item in res])
    else:
        res = [item['response'] for item in stream_jsonl(CHECKPOINTS['step 2'])]

    # Step 3: load pre-generated testcases and verify the samples
    slow_print('Solution get. Loading testcases...')
    if not os.path.exists(TEST_OUTPUT):
        sys.exit('Testcases file NOT FOUND.')
    else:
        testcase_list = [case for case in stream_jsonl(TEST_OUTPUT)]
    if not os.path.exists(CHECKPOINTS['step 3']):
        sample_list = match_solution_testcases(history, res, testcase_list)
        test_results = check_testcase(sample_list, inputs, verify=True, n_workers=16)
        write_jsonl(CHECKPOINTS['step 3'], test_results)
    else:
        test_results = [item for item in stream_jsonl(CHECKPOINTS['step 3'])]

    # Step 4: go over uni-agreement and request LLM for refinement
    # better_indices = uni_agreement(test_results, 3)  # find top k solution
    # output = [history[idx] for idx in better_indices]
    # step = next_step_prompts(history, test_result_1st, step)
    # sub_list = [{'index': line['index'], f'sub_prompt_{step}': line[f'sub_prompt_{step}']} for line in history if
    #             f'sub_prompt_{step}' in line]
    # slow_print(
    #     'self-refinement start.')  # this implementation only do self refinement once, multiple iterations are not implemented
    # res = service.request_response([line[f'sub_prompt_{step}'] for line in sub_list])
    # concatenate_str(sub_list, res, 'output', processing=True)
    # concatenate_dict(history, sub_list, ['output'], ['output'], has_indices=True)
    # write_jsonl(OUTPUTFILE, history)
    print('DONE')