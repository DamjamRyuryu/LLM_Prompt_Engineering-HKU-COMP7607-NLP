from utils import *
from baseline import write_jsonl, stream_jsonl, read_problems


OUTPUTFILE = "method_SelfEvolve.jsonl"
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

def get_input_list_se(evalset_file: str = HUMAN_EVAL):
    """
    the evaluation need task id as the anchor to relate question and completion, add it at first
    """
    _problems = read_problems(evalset_file)
    _prompts = []
    for p in _problems:
        _test = "assert".join(_problems[p]["test"].split('assert')[0:2])  # get the first test case
        _test = _test[_test.find("def"):]
        _prompts.append({'prompt':_problems[p]["prompt"], 'task_id':_problems[p]["task_id"], 'entry_point':_problems[p]["entry_point"], 'test':_test})
    return _prompts

def get_prompt_list_init(input_list: list[dict]):
    """
    the first step of SelfEvolve:
    construct the first prompt like this:
        (problem) + '\n' + FIRST_STEP['knowledge']
    it requires the model to generate knowledge (natural language) for solving the problem
    """
    prompt_lists = []
    for input_problem in input_list:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_problem['prompt'] + '\n' + FIRST_STEP['knowledge']}
        ]
        prompt_lists.append({'task_id': input_problem['task_id'],'prompt':prompt})
    return prompt_lists

def next_step_prompts(in_dict_list: list[dict], response: list[str | dict], step: int=0):
    """
    step 0 is completed in get_prompt_list_init
    Here handles the following steps:
        step 1: initial_prompt + response_1(appended after requesting response) + FIRST_STEP['solution']
        step 2: generated solution + SELF_REFINEMENT['syntax'/'error'] (constructed according to the test result)
    """
    assert len(in_dict_list) == len(response), f'Error: Wrong length of input lists when handling response. len(in) <{len(in_dict_list)}> should not be larger than len(append) <{len(response)}>'
    # construct chat history
    chat = [{'role':'assistant', 'content': content} for content in response]
    previous_prompt = [line[f'sub_prompt_{step}'] for line in in_dict_list]
    if step == 0:
        new_content = [previous_prompt[i] + [chat[i], {'role': 'user', 'content': FIRST_STEP['solution']}] for i in range(len(in_dict_list))]
        for i, line in enumerate(in_dict_list):
            line[f'sub_prompt_1'] = new_content[i]
    elif step == 1:
        for i, line in enumerate(in_dict_list):
            result = response[i]
            if result['passed']:
                line['prompt'] = line.pop(f'sub_prompt_{step}')
                line['output'] = result['completion']
            else:
                error_type = 'error' if result['result'] == "failed: " else 'syntax'
                # reconstruct the executed code
                executed_code = (
                    result['completion'] + "\n" +
                    result["test"] + "\n" +
                    f"check({result['entry_point']})"
                )
                if error_type == 'error':
                    append_content = SELF_REFINEMENT[error_type].split('%')
                    new_content = executed_code + '\n\n' + f"check({result['entry_point']})".join(append_content)
                else:
                    append_content = SELF_REFINEMENT[error_type].split('%')
                    new_content = executed_code + '\n\n' + result['result'].lstrip("failed: ").join(append_content)
                new_prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": new_content}
                ]
                line[f'sub_prompt_{step + 1}'] = new_prompt
                line['index'] = i  # record indices
    return step + 1

if __name__ == '__main__':
    service = LlamaModel(URL, API_KEY, 0.8, 1,0.8)
    inputs = get_input_list_se(HUMAN_EVAL)
    prompts = get_prompt_list_init(inputs)
    history = [{'task_id':item['task_id']} for item in inputs]
    concatenate_dict(history, inputs, ['input'], ['prompt'])
    concatenate_dict(history, prompts, ['sub_prompt_0'], ['prompt'])
    step = 0
    slow_print('first step start, ask for knowledge:')
    history = get_batch(history, 5)
    res = service.request_response([line[f'sub_prompt_{step}'] for line in history])
    slow_print('knowledge get, ask for solution:')
    step = next_step_prompts(history, res, step)
    res = service.request_response([line[f'sub_prompt_{step}'] for line in history])  # the response is solution now
    slow_print('check correctness with the first test case:')
    test_result_1st = check_testcase(
        [{'task_id': history[idx]['task_id'], 'completion': res[idx]} for idx in range(len(history))],
        inputs
    )
    step = next_step_prompts(history, test_result_1st, step)
    sub_list = [{'index': line['index'], f'sub_prompt_{step}': line[f'sub_prompt_{step}']} for line in history if f'sub_prompt_{step}' in line]
    slow_print('self-refinement start.')  # this implementation only do self refinement once, multiple iterations are not implemented
    res = service.request_response([line[f'sub_prompt_{step}'] for line in sub_list])
    concatenate_str(sub_list, res, 'output', processing=True)
    concatenate_dict(history, sub_list, ['output'], ['output'], has_indices=True)
    write_jsonl(OUTPUTFILE, history)
    print('DONE')