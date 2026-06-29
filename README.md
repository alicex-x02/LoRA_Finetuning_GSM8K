# Llama Fine-Tuning with LoRA on GSM8K

![Project Title](image/Title.png)

This project explores **LoRA fine-tuning for Llama-3.1-8B** on the **GSM8K mathematical reasoning dataset**.  
The main goal is to compare how different LoRA ranks and target modules affect training stability, validation loss, and final reasoning accuracy.

> For the full experimental setup, code details, logs, and analysis, see the project report:  
> [LoRA - Finetuning Project PDF](document/LoRA%20-%20Finetuning%20프로젝트.pdf)

---

## Overview

This experiment focuses on three main questions:

1. **LoRA Rank Comparison**
   - Compare low-rank LoRA (`r = 8`) and high-rank LoRA (`r = 64`).
   - Verify whether a larger rank always improves performance.

2. **Target Module Comparison**
   - Compare LoRA applied to:
     - Query & Value projections: `q_proj`, `v_proj`
     - Query, Key & Value projections: `q_proj`, `k_proj`, `v_proj`
     - All major modules
     - MLP modules

3. **Attention vs MLP Fine-Tuning**
   - Analyze whether applying LoRA to attention modules or MLP modules is more effective for mathematical reasoning.

---

## Experimental Setup

| Category | Setting |
|---|---|
| Base Model | `meta-llama/Llama-3.1-8B` |
| Dataset | GSM8K |
| GPU | NVIDIA V100 32GB |
| Framework | HuggingFace Transformers |
| Fine-tuning Method | LoRA |
| Precision | FP16 |
| Epochs | 3 |
| LoRA Ranks | `r = 8`, `r = 64` |

### Learning Rate

| Rank | Learning Rate |
|---|---|
| `r = 8` | `1e-4` |
| `r = 64` | `2e-5` |

A smaller learning rate was used for `r = 64` because the number of trainable parameters increases significantly, which can make training unstable.

---

## Repository Structure

```text
.
├── README.md
├── document/
│   └── LoRA - Finetuning 프로젝트.pdf
├── image/
│   ├── Title.png
│   └── graph.png
├── final_logs/
│   ├── r8_qv/
│   ├── r8_qkv/
│   ├── r8_all/
│   ├── r8_mlp/
│   ├── r64_qv/
│   ├── r64_qkv/
│   ├── r64_all/
│   └── r64_mlp/
├── inference_logs/
└── inference_mini_log/
