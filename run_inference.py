from openai import AsyncOpenAI
import pandas as pd
import asyncio
from tqdm import tqdm

client = AsyncOpenAI(base_url="http://localhost:8835/v1", api_key="EMPTY")

MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-14T23:11:17_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_sft/epoch_1/step_final"
# MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen25/exp_2025-09-15T00:08:26_qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_dft/epoch_1/step_final"

async def run(problem):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": problem}],
        temperature=1.0,
    )
    return response.choices[0].message.content

async def main():
    df = pd.read_csv('data/olympiad_bench.csv')[:5]
    questions = df['question'].tolist()

    model_responses = []
    for question_id in tqdm(range(0, len(questions), 10)):
        question_list = questions[question_id:question_id+10]
        tasks = [run(question) for question in question_list]
        responses = await asyncio.gather(*tasks)
        model_responses.extend(responses)

    print(questions[1])
    print(model_responses[1])
    print(df['final_answer'].iloc[1])
    print(df['solution'].iloc[1])
    df['model_response'] = model_responses
    df.to_csv('data/olympiad_bench_model_response.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())