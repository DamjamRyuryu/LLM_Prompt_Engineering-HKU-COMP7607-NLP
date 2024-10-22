from collections import Counter
from utils import *
from baseline import write_jsonl, stream_jsonl, read_problems
import os

OUTPUTFILE = "method_CodeT.jsonl"
TEST_OUTPUT = "pre_generated_data/generated_testcase.jsonl"
SYSTEM_PROMPT='''The user will ask you about Python code problem, follow their instructions. Pay attention to their required output format. Environment: ipython.'''
PROMPT_PADDING={
    'testcase': "For the above function, could you complete the test function (def check(candidate):) with 5 to 10 testcases?\nBe attention, \
you should only complete 'def check(candidate):' without any explanation, comment and natural language\n\
Don't provide very long assertation.\nDon't create new test function or class method.\nWarp your code with \"'''\"",
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
        prompt_t = [{"role": "user",
                     "content": input_problem['prompt'] + input_problem['ground_truth_fn']
                                + '\n' + input_problem['test'] + f'\n\ncheck({input_problem['entry_point']})'
                                + '\n\n' + PROMPT_PADDING['testcase']}]
        code_prompts.append({'task_id': input_problem['task_id'], 'prompt': prompt_0 + prompt_c})
        test_prompts.append({'task_id': input_problem['task_id'], 'prompt': prompt_0 + prompt_t})
    return {'solution':code_prompts,'testcase': test_prompts}

def construct_test_case(cases: list[str]):
    """
    put the assertations into the check function, output be like:
    def check(candidate):
        assert <test case>
    """
    _test = []
    for line in cases:
        _temp = "\ndef check(candidate):\n" + line
        _test.append(copy.deepcopy(_temp))
    return _test

def construct_verify_case(_testcases: list[dict]):
    """
    create verification lists for testing the generated testcases.
    use canonical_solution as the ground truth function
    """
    _verify_list = []
    for line in _testcases:
        for _case in line['testcase']:
            _temp_dict = {
                'task_id': line['task_id'],
                'completion': line['header'] + line['ground_truth_fn'],
                'test': _case,
                'entry_point': line['entry_point']
            }
            _verify_list.append(copy.deepcopy(_temp_dict))
    # delete duplicate tests
    _seen = set()
    _verify_sets = []
    for i, line in enumerate(_verify_list):
        _indicator = line['task_id'] + '-' +line['test'].strip()
        if _indicator not in _seen:
            _seen.add(_indicator)
            _verify_sets.append(line)
    return _verify_sets

def generate_test_cases(_input_file: list[dict], _prompts_file: list[dict], append: bool= False):
    """
    request the LLM to generate testcases for the problems
    the generated testcases will go through a check with the canonical solution to ensure its feasibility
    the checked testcases will be saved to TEST_OUTPUT.
    NO solution will be generated if TEST_OUTPUT doesn't exist, so rerun this script after generating the test file.
    """
    testcases = [
        {
            'task_id': item['task_id'],
            'header': item['prompt'],
            'entry_point': item['entry_point'],
            'ground_truth_fn': item['ground_truth_fn'],
            'template': item['test']
        } for item in _input_file
    ]
    concatenate_dict(testcases, _prompts_file, ['prompt'], ['prompt'])
    testcases = get_batch(testcases, 5)
    test_res = service.request_response([line['prompt'] for line in testcases])
    slow_print("Testcases generated, verifying with ground truth function...")
    concatenate_str(testcases, test_res, 'testcase')
    for sample in testcases:
        sample['testcase'] = construct_test_case(extract_assertation(sample['testcase']))
    single_tests = construct_verify_case(testcases)
    verify_results = check_testcase(single_tests, _input_file, verify=True)
    # clean dicts
    output = []
    for line in verify_results:
        if line['passed']:
            output.append({'task_id': line['task_id'], 'entry_point': line['entry_point'], 'test': line['test']})
    if append:
        slow_print(f"Result verified, {len(output)} testcases generated. Return the testcases")
        return output
    else:
        slow_print(f"Result verified, {len(output)} testcases generated. Saving the testcases...")
        write_jsonl(TEST_OUTPUT, output)
        slow_print("File saved. Please rerun the process.")
        return None

def count_testcase(_list : list[dict], _threshold: int):
    """
    count the testcases and return the 'task_id' indices having fewer cases than given _threshold
    used for finding problems without enough testcases
    """
    _keys = [d['task_id'] for d in _list]
    _counter = Counter(_keys)
    _filtered_list = [key for key, cnt in _counter.items() if cnt > 20]
    return _filtered_list


if __name__ == '__main__':
    service = LlamaModel(URL, API_KEY, 1.3, 1,1.0)
    inputs = get_input_list_ct(HUMAN_EVAL)
    prompts = get_prompt_list_ct(inputs)
    # history = [{'task_id': item['task_id'], 'input': item['prompt']} for item in inputs[0:10]]
    # # concatenate_dict(history, inputs, ['input'], ['prompt'])
    # concatenate_dict(history, prompts['solution'][0:10], ['prompt'], ['prompt'])
    # history = get_batch(history, 5)
    # slow_print('Prompts generate complete. Requesting for solutions...')
    # solutions = service.request_response([line['prompt'] for line in history])
    if not os.path.exists(TEST_OUTPUT):
        slow_print('No pre-generated testcases. Requesting for testcases...')
        generate_test_cases(inputs, prompts['testcase'])
    else:
        testcase = [case for case in stream_jsonl(TEST_OUTPUT)]
        lack_indices = count_testcase(testcase, 5)
        if lack_indices:
            slow_print('Some problems lack testcase, attempting to generate more...')
            lack_inputs = [line for line in inputs if line['task_id'] in lack_indices]
            lack_prompts = [line for line in prompts['testcase'] if line['task_id'] in lack_indices]
            supplied_case = generate_test_cases(lack_inputs, lack_prompts, append=True)
            testcase.extend(supplied_case)
            sorted_testcase = sorted(testcase, key=lambda d: int(d['task_id'].split('/')[1]))
            write_jsonl(TEST_OUTPUT.rstrip('.jsonl') + '_extra.jsonl',sorted_testcase)
            slow_print("Generate complete.")

    # write_jsonl(OUTPUTFILE, )
    print('DONE')