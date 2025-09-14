import datasets
import os

def load_numina_math():
    dataset = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train")
    dataset = dataset.filter(lambda x: x["source"] in ("synthetic_amc", "synthetic_math"))
    dataset = dataset.select_columns(["source", "problem", "solution"])
    dataset = dataset.to_pandas()
    return dataset

def load_math_500():
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select_columns(["problem", "solution", "answer"])
    dataset = dataset.to_pandas()
    return dataset

def load_minerva_math():
    dataset = datasets.load_dataset("math-ai/minervamath", split="test")
    dataset = dataset.select_columns(["question", "answer"])
    dataset = dataset.to_pandas()
    return dataset

def load_olympiad_bench():
    dataset = datasets.load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    dataset = dataset.select_columns(["question", "solution", "final_answer"])
    dataset = dataset.to_pandas()
    dataset['final_answer'] = dataset['final_answer'].apply(lambda x: x[0])
    return dataset

if __name__ == "__main__":
    dataset = load_numina_math()
    print(dataset.shape)
    # dataset = dataset.sample(50000, random_state=1023)
    # os.makedirs("data", exist_ok=True)
    # dataset.to_csv("data/numina_math.csv", index=False)

    # Testing for other datasets
    dataset = load_math_500()
    print(dataset.shape)
    dataset = load_minerva_math()
    print(dataset.shape)
    dataset = load_olympiad_bench()
    print(dataset.shape)