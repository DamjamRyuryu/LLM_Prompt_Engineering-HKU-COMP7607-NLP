The pipeline of this project is summarized as follows:

<font size =4>1. Result generation (Not applicable)</font>

Run one of the methods (baseline.py / self_evolve.py / codeT.py / combined.py)

Then you get the output of the method, which has already been included in the .zip file

<font size=1> This is not applicable because I removed the API key of the LLM. Provide your own key in "utils.py" if you want to generate results.</font>

<font size=4>2. Test the results </font>

Go to my_evaluation.py edit the "--output_file" to specify the result you want to examine.

If you refer running in terminal, run the original evaluation script (evaluate_functional_correctness.py) with command like:
    
    
    python evaluate_functional_correctness.py --sample_file pre_generated_data/HumanEval.jsonl --problem_file <result_file_path>
<font size=4>File structure</font>
    
    #######################     the original dataset and generated testcases
    --pre_generated_data/
        --generated_testcase.jsonl
        --HumanEval.jsonl
    #######################     integration of useful functions needed for multiple methods
    --utils.py
    #######################     testing scripts of HumanEval
    --evaluation.py
    --my_evaluation.py          (modified for running in PyCharm IDE)
    --evaluate_functional_correctness.py
    --modified_execution.py     (slightly different from original script)
    #######################     methods
    --baseline.py
    --self_evolve.py
    --codeT.py
    --combined.py
    #######################     output files(fit with the required format in email)
    --zeroshot_baseline.jsonl
    --method_SelfEvolve.jsonl
    --method_CodeT.jsonl
    --method_combined.jsonl
    