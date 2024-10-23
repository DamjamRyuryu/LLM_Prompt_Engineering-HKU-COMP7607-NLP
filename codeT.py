from utils import *
from baseline import write_jsonl, stream_jsonl, read_problems
import os

BATCH_SIZE = 5
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

def construct_test_case(cases: list[str]) -> list[str]:
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

def construct_verify_case(_testcases: list[dict]) -> list[dict]:
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

def count_testcase(_list : list[dict], _threshold: int) -> list[str]:
    """
    count the testcases and return the 'task_id' indices having fewer cases than given _threshold
    used for finding problems without enough testcases
    """
    _keys = [d['task_id'] for d in _list]
    _counter = Counter(_keys)
    _filtered_list = [key for key, cnt in _counter.items() if cnt < _threshold]
    return _filtered_list

def match_solution_testcases(_inputs: list[dict], _solutions: list[str], _cases: list[dict]) -> list[dict]:
    """
    create verify list with solutions and testcases
    """
    for i, line in enumerate(_solutions):
        header = _inputs[i]['input'][:_inputs[i]['input'].find('def ')].strip()  # get the imported modules before the function in case the solution omits them
        line = solution_to_completion(line)
        if not header in line[:line.find('def ')].strip():
            _completion = header + '\n\n' + line
        else:
            _completion = line
        _inputs[i]['output'] = _completion
    # construct verify sets
    _v_list = []
    for i, line in enumerate(_inputs):
        task_id = line['task_id']
        _set = [{
            'task_id': task_id, 'completion': line['output'], 'test': item['test'], 'entry_point': item['entry_point'], 'line_index': i
        } for item in _cases if item['task_id'] == task_id]
        _v_list.extend(copy.deepcopy(_set))
    return  _v_list

def uni_agreement(_results: list[dict], k: int=3) -> list[int]:
    """
    find the top k solutions according to their verification results on testcases
    return a list of integers indicating the line index of top k results in the whole solution list
    """
    assert BATCH_SIZE >= k, 'ERROR: not enough batches'
    _counter = defaultdict(Counter)
    # count passed cases for single solution
    for line in _results:
        if line['passed']:
            _counter[line['task_id']][line['line_index']] += 1
    # choose top k solutions
    _indices = []
    for task_id, line_index in _counter.items():
        if len(line_index) >= k:
            top_k = line_index.most_common(k)
            _indices.extend(copy.deepcopy([i[0] for i in top_k]))
        elif len(line_index) > 0:
            _diff = k - len(line_index)
            top_k = line_index.most_common(len(line_index))
            _idx_start = int(task_id.split('/')[1]) * BATCH_SIZE
            _idx_group = [_idx_start + i for i in range(BATCH_SIZE)]
            correct_cases = [i[0] for i in top_k]
            left_cases = [i for i in _idx_group if i not in correct_cases]
            _indices.extend(correct_cases + left_cases[0:_diff])
        else:
            slow_print(f'Warning: task_id {task_id}, has no correct solution')
            _idx_start = int(task_id.split('/')[1]) * BATCH_SIZE
            _append_indices = [_idx_start + i for i in range(k)]
            _indices.extend(_append_indices)

    return _indices

if __name__ == '__main__':
    service = LlamaModel(URL, API_KEY, 0.8, 1,0.8)
    inputs = get_input_list_ct(HUMAN_EVAL)
    prompts = get_prompt_list_ct(inputs)
    if not os.path.exists(TEST_OUTPUT):
        slow_print('No pre-generated testcases. Requesting for testcases...')
        generate_test_cases(inputs, prompts['testcase'])
        testcase_list = []
    else:
        testcase_list = [case for case in stream_jsonl(TEST_OUTPUT)]
        # # supply problems with more testcases. already done so skip the procedure
        # lack_indices = count_testcase(testcase, 5)
        # if lack_indices:
        #     slow_print('Some problems lack testcase, attempting to generate more...')
        #     lack_inputs = [line for line in inputs if line['task_id'] in lack_indices]
        #     lack_prompts = [line for line in prompts['testcase'] if line['task_id'] in lack_indices]
        #     supplied_case = generate_test_cases(lack_inputs, lack_prompts, append=True)
        #     testcase.extend(supplied_case)
        #     sorted_testcase = sorted(testcase, key=lambda d: int(d['task_id'].split('/')[1]))
        #     write_jsonl(TEST_OUTPUT.rstrip('.jsonl') + '_extra.jsonl',sorted_testcase)
        slow_print("Testcases loaded.")
    if testcase_list:
        history = [{'task_id': item['task_id'], 'input': item['prompt']} for item in inputs]
        concatenate_dict(history, prompts['solution'], ['prompt'], ['prompt'])
        history = get_batch(history, BATCH_SIZE)
        slow_print('Prompts generate complete. Requesting for solutions...')
        if os.path.exists('CodeT_response_temp.jsonl'):
            solutions = [item['response'] for item in stream_jsonl('CodeT_response_temp.jsonl')]
        else:
            solutions = service.request_response([line['prompt'] for line in history])
            # log the response so that the results can be tested later (the whole process will be time-consuming)
            write_jsonl('CodeT_response_temp.jsonl', [{'response': item} for item in solutions])
        slow_print('check correctness with the testcases...')
        sample_list = match_solution_testcases(history, solutions, testcase_list)
        if not os.path.exists('ckpt_codeT.jsonl'):
            test_results = check_testcase(sample_list, inputs, verify=True)
            write_jsonl('ckpt_codeT.jsonl', test_results)
        else:
            test_results = [item for item in stream_jsonl('ckpt_codeT.jsonl')]
        better_indices = uni_agreement(test_results, 3)  # find top k solution
        output = [history[idx] for idx in better_indices]
        write_jsonl(OUTPUTFILE, output)
    print('DONE')