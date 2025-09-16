## Overview

This repo contains a re-implementation of the DFT paper described in [https://arxiv.org/abs/2508.05629](https://arxiv.org/abs/2508.05629). We train the Qwen2.5-Math-1.5B on NuminaMath-CoT and evaluate on three math benchmarks. Two training losses are implemented:
- Standard SFT cross-entropy
- DFT (dynamically-reweighted SFT)

### DFT loss (dynamically-reweighted SFT)

Let $p_θ(y_t | x)$ be the model probability of the gold token $y_t$ at position $t$. Standard SFT minimizes $−\log p_θ(y_t | x)$. DFT instead minimizes

- per-token: $L_t = − p_θ(y_t | x) · \log p_θ(y_t | x)$
- sequence loss: mean over non-ignored tokens

Equivalently, DFT multiplies the CE at each token by the model’s own confidence on the gold token. This down-weights gradients where the model is very uncertain (small p) while still encouraging confidence improvement; when p is high, it behaves similarly to CE but with a softer slope. Implementation: `utils/loss_fn.py::DFTForCausalLMLoss` computes per-token CE with `reduction="none"`, gathers p_θ(y_t | x) from the softmax, multiplies, masks ignore indices, and normalizes by token count (or batch items when provided).

Differences from SFT:
- SFT: $L_t = −\log p_θ(y_t | x)$
- DFT: $L_t = −p_θ(y_t | x) \log p_θ(y_t | x)$

This can be seen as emphasizing tokens the model already puts some mass on, reducing the harsh penalty on hard tokens early in training.

## Methodology

### Training data
- Source: `AI-MO/NuminaMath-CoT` filtered to `source ∈ {synthetic_amc, synthetic_math}` via `load_data.py`.
- Preprocessing: `prepare_training_data.py` converts to chat-format JSONL using the Qwen chat template (`instruction` + `output`).

Minimal repro:

```bash
python load_data.py            # writes CSVs under data/
python prepare_training_data.py  # writes JSONLs under data/
```

### Training
- Model: `Qwen/Qwen2.5-Math-1.5B`, FlashAttention-2, FSDP, optional torch.compile.
- Loss: choose with `--loss-type {sft,dft}`.

```bash
bash bash_train_qwen3_flash_fsdp_compile_dcp.sh
# or directly
torchrun --nnodes=1 --nproc-per-node=4 --master_port 29518 run_train_qwen3_fsdp.py --dcp-api --loss-type dft
```

### Evaluation methodology
We evaluate on three benchmarks after training on NuminaMath-CoT:
- Math 500 (`HuggingFaceH4/MATH-500`)
- Minerva Math (`math-ai/minervamath`)
- Olympiad Bench (`Hothan/OlympiadBench`, subset `OE_TO_maths_en_COMP`)

Answer extraction and grading:
- Extract answers with `utils/parse_answer.extract_answer` (handles boxed, LaTeX, last-number heuristic, etc.).
- Compare with `utils/grader.math_equal`, which checks numeric closeness, symbolic equality via SymPy/latex2sympy, matrices, equations, and simple choice answers.

Parsing and grading code from the official repository: [https://github.com/yongliang-wu/DFT](https://github.com/yongliang-wu/DFT)

Protocols reported below:
- Temperature 0.0 single pass.
- Temperature 0.2, best-of-16: run 16 independent samples, majority-vote the final answers per question.

Serving and evaluation commands:

```bash
# Serve the finetuned model with vLLM (OpenAI-compatible API)
bash run_vllm_server.sh  # default port 8836

# Run batched evaluation against the server (uses port 8836)
python run_evaluation.py

# (Optional) Cache raw generations and score offline
python run_inference.py   # writes CSVs under data/
python score.py           # prints accuracy per dataset
```

## Results

> I was not able to reproduce the pattern of results in the paper. The performance of the model trained with the DFT loss is lower than the model trained with the standard SFT loss. One possibility could be because I used less training data (only the synthetic_amc and synthetic_math subsets). Another thing I did note however is that the Numina-Math dataset mentioned in the paper already potentially includes other math benchmark data, so I'm not sure if the apporoach mentioned in the paper is correct.

### Temperature = 0

|     | Math 500 | Minerva Math | Olympiad Bench |
|-----|----------|--------------|----------------|
| SFT | 0.624    | 0.264        | 0.284          |
| DFT | 0.63     | 0.213        | 0.253          |

### Temperature = 0.2; Best of 16 runs

|     | Math 500 | Minerva Math | Olympiad Bench |
|-----|----------|--------------|----------------|
| SFT | 0.694    | 0.286        | 0.320          |
| DFT | 0.648    | 0.217        | 0.262          |