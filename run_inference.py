from openai import AsyncOpenAI
import pandas as pd
import asyncio
from tqdm import tqdm

client = AsyncOpenAI(base_url="http://localhost:8835/v1", api_key="EMPTY")

# LOSS_TYPE = "sft"
# MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-14T23:11:17_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_sft/epoch_1/step_final"

LOSS_TYPE = "dft"
MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-15T00:08:26_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_dft/epoch_1/step_final"

async def run(problem):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": problem}],
        temperature=0.0,
    )
    return response.choices[0].message.content

async def run_batch(questions):
    model_responses = []
    for question_id in tqdm(range(0, len(questions), 10)):
        question_list = questions[question_id:question_id+10]
        tasks = [run(question) for question in question_list]
        responses = await asyncio.gather(*tasks)
        model_responses.extend(responses)
    return model_responses

async def run_math_500():
    df = pd.read_csv(f'data/math_500.csv')
    questions = df['problem'].tolist()

    model_responses = await run_batch(questions)
    df['model_response'] = model_responses
    df.to_csv(f'data/math_500_model_response_{LOSS_TYPE}.csv', index=False)

async def run_minerva_math():
    df = pd.read_csv(f'data/minerva_math.csv')
    questions = df['question'].tolist()

    model_responses = await run_batch(questions)
    df['model_response'] = model_responses
    df.to_csv(f'data/minerva_math_model_response_{LOSS_TYPE}.csv', index=False)

async def run_olympiad_bench():
    df = pd.read_csv(f'data/olympiad_bench.csv')
    questions = df['question'].tolist()

    model_responses = await run_batch(questions)
    df['model_response'] = model_responses
    df.to_csv(f'data/olympiad_bench_model_response_{LOSS_TYPE}.csv', index=False)

if __name__ == "__main__":
    print("Running inference for math_500")
    asyncio.run(run_math_500())
    print("Running inference for minerva_math")
    asyncio.run(run_minerva_math())
    print("Running inference for olympiad_bench")
    asyncio.run(run_olympiad_bench())