import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import json
from tqdm import tqdm

tqdm.pandas()

def convert_to_jsonl(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": row["problem"]
        }
    ]
    instruction = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = row["solution"] + tokenizer.eos_token
    
    return {"instruction": instruction, "output": output}

def dump_to_jsonl(records, path):
    with open(path, 'w') as outfile:
        for entry in records:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    df = pd.read_csv("data/numina_math.csv")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    train_data = df_train.progress_apply(convert_to_jsonl, axis=1, tokenizer=tokenizer)
    test_data = df_test.progress_apply(convert_to_jsonl, axis=1, tokenizer=tokenizer)

    dump_to_jsonl(train_data, "data/numina_math_train.jsonl")
    dump_to_jsonl(test_data, "data/numina_math_test.jsonl")
