from utils import *
from baseline import write_jsonl, stream_jsonl, read_problems
import os

MIN_CASES = 25
OUTPUTFILE = "method_CodeT.jsonl"
TEST_OUTPUT = "pre_generated_data/generated_testcase.jsonl"
SYSTEM_PROMPT='''The user will ask you about Python code problem, follow their instructions. Pay attention to their required output format. Environment: ipython.'''
PROMPT_PADDING={
    'testcase': "For the above function, could you provide 5 or more testcases for the test function (check(candidate))?\nBe attention, \
you should only output the codes for the test function without any explanation, comment, natural language\n\
Warp your code with \"'''\"",
    'solution': "Help me complete the function.\nBe attention, you should only output the codes without any explanation, comment, natural language and testcode.\n\
Warp your code with \"'''\""
}

def get_input_list_ct(evalset_file: str = HUMAN_EVAL):
    _problems = read_problems(evalset_file)
    _prompts = []
    for p in _problems:
        _test = _problems[p]["test"].split('assert')[0] + 'assert'  # get the test function and the first assert as a hint
        _test = _test[_test.find("def"):]
        _prompts.append({'prompt': _problems[p]["prompt"], 'task_id': _problems[p]["task_id"],
                         'entry_point': _problems[p]["entry_point"], 'ground_truth_fn': _problems[p]['canonical_solution'], 'test': _test})
    return _prompts


def get_prompt_list_ct(input_list):
    code_prompts = []
    test_prompts = []
    for input_problem in input_list:
        prompt_0 = [{"role": "system", "content": SYSTEM_PROMPT}]
        prompt_c = [{"role": "user", "content": input_problem['prompt'] + '\n' + PROMPT_PADDING['solution']}]
        prompt_t = [{"role": "user", "content": input_problem['prompt'] + '\n' + PROMPT_PADDING['testcase']}]
        code_prompts.append({'task_id': input_problem['task_id'], 'prompt': prompt_0 + prompt_c})
        test_prompts.append({'task_id': input_problem['task_id'], 'prompt': prompt_0 + prompt_t})
    return {'solution':code_prompts,'testcase': test_prompts}


if __name__ == '__main__':
    service = LlamaModel(URL, API_KEY, 1.0, 1,1.0)
    inputs = get_input_list_ct(HUMAN_EVAL)
    prompts = get_prompt_list_ct(inputs)
    history = [{'task_id': item['task_id'], 'input': item['prompt']} for item in inputs[0:10]]
    # concatenate_dict(history, inputs, ['input'], ['prompt'])
    concatenate_dict(history, prompts['solution'][0:10], ['prompt'], ['prompt'])
    history = get_batch(history, 5)
    slow_print('Prompts generate complete. Requesting for solutions...')
    solutions = service.request_response([line['prompt'] for line in history])
    if not os.path.exists(TEST_OUTPUT):
        slow_print('No pre-generated testcases. Requesting for testcases...')
        testcases = [
            {
                'task_id': item['task_id'],
                'entry_point': item['entry_point'],
                'ground_truth_fn': item['ground_truth_fn'],
                'template': item['test']
            } for item in inputs
        ]
        concatenate_dict(testcases, prompts['testcase'], ['prompt'], ['prompt'])
        test_res = service.request_response([line['prompt'] for line in testcases])
        concatenate_str(testcases, test_res, 'testcase')
        pass
    # write_jsonl(OUTPUTFILE, )
    print('DONE')