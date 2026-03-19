# Prompting-Based Lead Optimization with Open LLMs

**Author:** Ziqing Wang
**Course:** STAT 359, Northwestern University
**Date:** March 2026

## Overview

This project systematically evaluates whether advanced prompting strategies (zero-shot, few-shot, chain-of-thought) can enable general-purpose LLMs to perform competitive molecular lead optimization at inference time, without any task-specific fine-tuning.

We compare two general-purpose LLMs (**Llama-3.1-8B**, **Qwen2.5-7B**) against five task-specific SFT models (**ChemLLM**, **LlaSMol**, **PEIT-LLM**, **DrugAssist**, **GeLLMO**) across five pharmacologically relevant properties using four evaluation metrics.

## Project Structure

```
student/final_project/
├── lead_optimization.py   # Main evaluation script
├── zinc_test_200.json     # 200 test molecules sampled from ZINC
└── README.md              # This file
```

## Dependencies

```
torch
vllm
transformers
llamafactory
rdkit
tdc
numpy
tqdm
```

Install with:
```bash
pip install torch vllm transformers rdkit-pypi PyTDC numpy tqdm
# LLaMA-Factory: follow https://github.com/hiyouga/LLaMA-Factory
```

## Data

**`zinc_test_200.json`** — 200 molecules sampled from the ZINC database. Each entry contains:
- `source_smiles`: input SMILES string
- `task`: target property (e.g., `"qed"`)
- `properties`: source property values

## Usage

### General-Purpose LLMs (Prompting Experiments)

```bash
# Zero-shot evaluation with Llama-3.1-8B
python lead_optimization.py \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --template llama3 \
  --prompt_type general \
  --prompting_strategy zero-shot \
  --task qed \
  --test_data_path zinc_test_200.json \
  --num_molecules 200 \
  --similarity_threshold 0.6 \
  --temperature 0.0 \
  --output_dir ./results/llama_zeroshot

# Few-shot (3 demonstration examples)
python lead_optimization.py \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --template llama3 \
  --prompt_type general \
  --prompting_strategy few-shot \
  --task qed \
  --test_data_path zinc_test_200.json \
  --similarity_threshold 0.6 \
  --temperature 0.0 \
  --output_dir ./results/llama_fewshot

# Chain-of-thought
python lead_optimization.py \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --template llama3 \
  --prompt_type general \
  --prompting_strategy cot \
  --task qed \
  --test_data_path zinc_test_200.json \
  --similarity_threshold 0.6 \
  --temperature 0.0 \
  --output_dir ./results/llama_cot

# Qwen2.5-7B (same flags, different model)
python lead_optimization.py \
  --model_path /path/to/Qwen2.5-7B-Instruct \
  --template qwen2 \
  --prompt_type general \
  --prompting_strategy zero-shot \
  --task qed \
  --test_data_path zinc_test_200.json \
  --similarity_threshold 0.6 \
  --temperature 0.0 \
  --output_dir ./results/qwen_zeroshot
```

### SFT Baseline Models

```bash
# DrugAssist
python lead_optimization.py \
  --model_path /path/to/DrugAssist \
  --template drugassist \
  --prompt_type sft \
  --task qed \
  --test_data_path zinc_test_200.json \
  --similarity_threshold 0.6 \
  --temperature 0.0 \
  --output_dir ./results/drugassist

# GeLLMO (with LoRA adapter)
python lead_optimization.py \
  --model_path /path/to/base_model \
  --adapter_path /path/to/gellmo_lora \
  --template llama3 \
  --prompt_type sft \
  --task qed \
  --test_data_path zinc_test_200.json \
  --similarity_threshold 0.6 \
  --output_dir ./results/gellmo
```

### Other Target Properties

Replace `--task qed` with any of: `logp`, `jnk3`, `gsk3b`, `drd2`.

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompting_strategy` | `zero-shot` | `zero-shot`, `few-shot`, or `cot` |
| `--prompt_type` | `sft` | `sft` (fine-tuned models) or `general` (prompting targets) |
| `--task` | (required) | Target property: `qed`, `logp`, `jnk3`, `gsk3b`, `drd2` |
| `--similarity_threshold` | `0.6` | Minimum Tanimoto similarity to source molecule |
| `--temperature` | `0.7` | Sampling temperature (use `0.0` for greedy decoding) |
| `--num_molecules` | `200` | Number of test molecules |
| `--target_oracle_calls` | `500` | Oracle call budget per molecule |

## Evaluation Metrics

The four evaluation dimensions (from report Section 3.4):

| Metric | Description |
|---|---|
| **Validity (Val)** | Fraction of outputs parsable by RDKit as valid SMILES |
| **Property Control (PC)** | Fraction where target property improved AND Tanimoto >= 0.6 |
| **Synthesizability (Syn)** | Mean SA Score of valid molecules (1-10; lower is better) |
| **Diversity (Div)** | Internal diversity: 1 - mean(pairwise Tanimoto) of valid outputs |

## Output

Results are saved as JSON files in `--output_dir` with:
- Per-molecule results (source SMILES, best candidate, improvements, similarity)
- Aggregate metrics (Val, PC, Syn, Div, success rates)
- Runtime and configuration metadata

## Individual Contributions

This project was completed individually by **Ziqing Wang**, who designed the experimental framework, implemented the prompting strategies, ran all evaluations, and wrote the report.

## License

MIT
