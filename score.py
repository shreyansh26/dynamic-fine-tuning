import pandas as pd
from latex2sympy2 import latex2sympy, latex2latex
from utils.parse_answer import extract_answer
from utils.grader import math_equal
from tqdm import tqdm

tqdm.pandas()

def evaluate(model_response, gt_answer, data_name):
    model_answer = extract_answer(model_response, data_name)
    # print(model_answer)
    # print(gt_answer)
    is_correct = math_equal(model_answer, gt_answer, timeout=True)
    return is_correct

def score_math_500(data_name, loss_type):
    df = pd.read_csv(f"data/math_500_model_response_{loss_type}.csv")
    correct_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        model_response = row["model_response"]
        gt_answer = row["answer"]
        is_correct = evaluate(model_response, gt_answer, data_name)
        if is_correct:
            correct_count += 1
    print(f"Data name: {data_name}")
    print(f"Loss type: {loss_type}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(df)}")
    print(f"Accuracy: {correct_count/len(df)}")

def score_minerva_math(data_name, loss_type):
    df = pd.read_csv(f"data/minerva_math_model_response_{loss_type}.csv")
    correct_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        model_response = row["model_response"]
        gt_answer = row["answer"]
        is_correct = evaluate(model_response, gt_answer, data_name)
        if is_correct:
            correct_count += 1
    print(f"Data name: {data_name}")
    print(f"Loss type: {loss_type}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(df)}")
    print(f"Accuracy: {correct_count/len(df)}")

def score_olympiad_bench(data_name, loss_type):
    df = pd.read_csv(f"data/olympiad_bench_model_response_{loss_type}.csv")
    correct_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        model_response = row["model_response"]
        gt_answer = row["final_answer"]
        is_correct = evaluate(model_response, gt_answer, data_name)
        if is_correct:
            correct_count += 1
    print(f"Data name: {data_name}")
    print(f"Loss type: {loss_type}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(df)}")
    print(f"Accuracy: {correct_count/len(df)}")

if __name__ == "__main__":
    score_math_500("math_500", "sft")
    score_math_500("math_500", "dft")

    score_minerva_math("minerva_math", "sft")
    score_minerva_math("minerva_math", "dft")

    score_olympiad_bench("olympiad_bench", "sft")
    score_olympiad_bench("olympiad_bench", "dft")