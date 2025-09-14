import datasets
import os

def load_numina_math():
    dataset = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train")
    dataset = dataset.filter(lambda x: x["source"] in ("synthetic_amc", "synthetic_math"))
    dataset = dataset.select_columns(["source", "problem", "solution"])
    dataset = dataset.to_pandas()
    return dataset

if __name__ == "__main__":
    dataset = load_numina_math()
    dataset = dataset.sample(50000, random_state=1023)
    os.makedirs("data", exist_ok=True)
    dataset.to_csv("data/numina_math_50k.csv", index=False)