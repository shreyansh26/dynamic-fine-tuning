from openai import AsyncOpenAI
import pandas as pd
import asyncio
from tqdm import tqdm
from utils.parse_answer import extract_answer
from utils.grader import math_equal
from collections import Counter

client = AsyncOpenAI(base_url="http://localhost:8836/v1", api_key="EMPTY")

# LOSS_TYPE = "sft"
# MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-14T23:11:17_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_sft/epoch_1/step_final"

LOSS_TYPE = "dft"
MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-15T00:08:26_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_dft/epoch_1/step_final"

async def run(problem, temperature=0.0):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": problem}],
        temperature=temperature,
    )
    return response.choices[0].message.content

async def run_batch(questions, temperature=0.0):
    model_responses = []
    for question_id in tqdm(range(0, len(questions), 10)):
        question_list = questions[question_id:question_id+10]
        tasks = [run(question, temperature) for question in question_list]
        responses = await asyncio.gather(*tasks)
        model_responses.extend(responses)
    return model_responses

async def evaluate_math_500(num_runs=16):
    df = pd.read_csv(f'data/math_500.csv')
    questions = df['problem'].tolist()

    model_answers_all_runs = []
    for i in range(num_runs):
        model_responses = await run_batch(questions, temperature=0.2)
        model_answers = [extract_answer(model_response, 'math_500') for model_response in model_responses]
        model_answers_all_runs.append(model_answers)

    correct_count = 0
    for question_id in range(len(questions)):
        answers = []
        for run_id in range(num_runs):
            answers.append(model_answers_all_runs[run_id][question_id])
        answers_frequency = Counter(answers)
        answers_frequency = sorted(answers_frequency.items(), key=lambda x: x[1], reverse=True)
        final_answer = answers_frequency[0][0]
        gt_answer = df.at[question_id, 'answer']
        is_correct = math_equal(final_answer, gt_answer, timeout=True)
        if is_correct:
            correct_count += 1
    print(f"Data name: math_500")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(questions)}")
    print(f"Accuracy: {correct_count/len(questions)}")

async def evaluate_minerva_math(num_runs=16):
    df = pd.read_csv(f'data/minerva_math.csv')
    questions = df['question'].tolist()

    model_answers_all_runs = []
    for i in range(num_runs):
        model_responses = await run_batch(questions, temperature=0.2)
        model_answers = [extract_answer(model_response, 'minerva_math') for model_response in model_responses]
        model_answers_all_runs.append(model_answers)

    correct_count = 0
    for question_id in range(len(questions)):
        answers = []
        for run_id in range(num_runs):
            answers.append(model_answers_all_runs[run_id][question_id])
        answers_frequency = Counter(answers)
        answers_frequency = sorted(answers_frequency.items(), key=lambda x: x[1], reverse=True)
        final_answer = answers_frequency[0][0]
        gt_answer = df.at[question_id, 'answer']
        is_correct = math_equal(final_answer, gt_answer, timeout=True)
        if is_correct:
            correct_count += 1
    print(f"Data name: minerva_math")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(questions)}")
    print(f"Accuracy: {correct_count/len(questions)}")

async def evaluate_olympiad_bench(num_runs=16):
    df = pd.read_csv(f'data/olympiad_bench.csv')
    questions = df['question'].tolist()

    model_answers_all_runs = []
    for i in range(num_runs):
        model_responses = await run_batch(questions, temperature=0.2)
        model_answers = [extract_answer(model_response, 'olympiad_bench') for model_response in model_responses]
        model_answers_all_runs.append(model_answers)

    correct_count = 0
    for question_id in range(len(questions)):
        answers = []
        for run_id in range(num_runs):
            answers.append(model_answers_all_runs[run_id][question_id])
        answers_frequency = Counter(answers)
        answers_frequency = sorted(answers_frequency.items(), key=lambda x: x[1], reverse=True)
        final_answer = answers_frequency[0][0]
        gt_answer = df.at[question_id, 'final_answer']
        is_correct = math_equal(final_answer, gt_answer, timeout=True)
        if is_correct:
            correct_count += 1
    print(f"Data name: olympiad_bench")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"Correct count: {correct_count}")
    print(f"Total count: {len(questions)}")
    print(f"Accuracy: {correct_count/len(questions)}")

if __name__ == "__main__":
    print("Running evaluation for math_500")
    asyncio.run(evaluate_math_500())

    print("Running evaluation for minerva_math")
    asyncio.run(evaluate_minerva_math())

    print("Running evaluation for olympiad_bench")
    asyncio.run(evaluate_olympiad_bench())