# Finetuning Large Language Models - DeepLearning.AI Project

> GitHub README for the project derived from the DeepLearning.AI course notebooks: `Training.ipynb` and `Evaluation.ipynb`.

---

## Project overview

This repository contains two Jupyter notebooks that implement and demonstrate a practical finetuning workflow for large language models (LLMs) following the DeepLearning.AI finetuning curriculum:

- `Training.ipynb` — notebook that contains the full finetuning pipeline: data loading & preprocessing, model selection/wrapping, training loop (or trainer usage), checkpointing, and logging.
- `Evaluation.ipynb` — notebook that loads checkpoints and performs qualitative and quantitative evaluation: generation examples, automated metrics, and basic error analysis.

The goal of this project is to provide a reproducible, clear demonstration of finetuning an LLM for a task (e.g., instruction-following, summarization, classification, or question-answering) and to document best practices used in the DeepLearning.AI course.

---

## Features

- Notebook-driven, reproducible finetuning workflow.
- Support for Hugging Face Transformers model families (GPT-style / causal; encoder-decoder optional).
- Tokenization and dataset preprocessing utilities.
- Checkpoint saving and resumable training.
- Evaluation utilities including generation examples and automatic metrics (e.g., BLEU, ROUGE, accuracy where appropriate).
- Clear logging and experiment notes for easy replication.

---

## Repository structure

```
/  (root)
├─ Training.ipynb         # Finetuning pipeline
├─ Evaluation.ipynb       # Evaluation and analysis
├─ requirements.txt       # Python dependencies
├─ data/                  # Example dataset location (not included)
├─ checkpoints/           # Saved model checkpoints
├─ notebooks/             # (optional) helper notebooks
└─ README.md              # This file
```

---

## Requirements

Recommended environment (adjust to your setup):

- Python 3.9+ (3.10 recommended)
- Jupyter / JupyterLab
- PyTorch (compatible version for your CUDA or CPU)
- Hugging Face Transformers
- Datasets (Hugging Face)
- accelerate (optional — for multi-GPU / distributed)
- bitsandbytes (optional — for 8-bit training)
- Additional packages used in notebooks (pandas, tqdm, rouge_score, sacrebleu, etc.)

A sample `requirements.txt` entry:

```
transformers>=4.30
datasets>=2.10
torch>=2.0
accelerate
bitsandbytes
evaluate
pandas
tqdm
sacrebleu
rouge-score
jupyterlab
```

Install with:

```bash
python -m pip install -r requirements.txt
```

---

## How to use

### 1) Open the notebooks

Open `Training.ipynb` and `Evaluation.ipynb` in Jupyter or JupyterLab. Each notebook is annotated and broken into sections for clarity.

### 2) Configure dataset & paths

At the top of `Training.ipynb` you will find configuration variables (or a `config` cell). Set these before running the notebook:

- `DATA_PATH` — path to your training dataset (or HF dataset id).
- `OUTPUT_DIR` — directory to save checkpoints and logs.
- `MODEL_NAME` — pretrained model checkpoint to finetune (e.g., `gpt2`, `gpt-neo-1.3B`, or `meta-llama/Llama-2-7b` if you have access and resources).
- `BATCH_SIZE`, `LR`, `EPOCHS`, `MAX_LENGTH`, etc.

Example configuration cell (already present in the notebook):

```python
DATA_PATH = "data/my_dataset.jsonl"
OUTPUT_DIR = "checkpoints/exp1"
MODEL_NAME = "gpt2"
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 3
MAX_LENGTH = 512
```

> Note: Large models may require `accelerate` and specialized hardware. If using 8-bit training or gradient checkpointing, the notebook contains optional cells to enable those.

### 3) Run training

- Run the cells in `Training.ipynb` sequentially. The notebook covers dataset preprocessing, tokenization, Trainer/loop setup, and the training run.
- If you prefer a script-based run (recommended for long runs), the notebook includes a code snippet that can be adapted into a `train.py` and executed with `accelerate launch`.

Example (script-style) command:

```bash
accelerate launch train.py --config configs/exp1.yaml
```

### 4) Checkpoints and logs

Trained checkpoints and logs are saved under `OUTPUT_DIR`. Use the saved model files or the `.bin`/`pytorch_model.bin` files with `transformers` `from_pretrained`.

### 5) Evaluation

Open `Evaluation.ipynb` and point it to the same `OUTPUT_DIR` or a specific checkpoint. Run the evaluation cells to generate examples (few-shot prompts, zero-shot), compute automatic metrics, and inspect failure cases.

---

## Example prompts & evaluation

`Evaluation.ipynb` contains example prompts and generation configurations (temperature, top_k, top_p, max_length). It also includes automated scripts to compute metrics like ROUGE and BLEU for text-to-text tasks and accuracy for classification-like tasks.

Suggested generation config:

```python
gen_config = {
    "max_new_tokens": 128,
    "temperature": 0.2,
    "top_p": 0.95,
    "do_sample": True
}
```

---

## Hyperparameters & tips

- Start with small learning rates for pretrained LLMs (e.g., `1e-5`—`5e-5`).
- Use gradient accumulation to simulate larger batch sizes when GPU memory is limited.
- Consider mixed precision (`fp16`) or 8-bit optimizers (bitsandbytes) to speed up training and reduce memory.
- Save checkpoints frequently early on, but keep a validation checkpointing schedule to reduce storage.
- Use a held-out validation set for early stopping and to choose best checkpoints.

---

## Troubleshooting

- `OutOfMemoryError`: reduce batch size, enable gradient checkpointing, or use 8-bit training.
- Slow training: enable mixed precision, use a smaller model for iteration, or use `accelerate` to distribute training.
- Tokenization issues: ensure the tokenizer and model match (same pretrained name) and the dataset is cleaned of unexpected tokens/newlines.


---

## Acknowledgements

This project follows the DeepLearning.AI course on finetuning LLMs and uses the Hugging Face Transformers & Datasets ecosystem.


