# Evaluation

Llama-Recipe make use of `lm-evaluation-harness` for evaluating our fine-tuned Meta Llama3 (or Llama2) model. It also can serve as a tool to evaluate quantized model to ensure the quality in lower precision or other optimization applied to the model that might need evaluation.

`lm-evaluation-harness` provide a wide range of [features](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#overview):

- Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via transformers (including quantization via AutoGPTQ), GPT-NeoX, and Megatron-DeepSpeed, with a flexible tokenization-agnostic interface.
- Support for fast and memory-efficient inference with vLLM.
- Support for commercial APIs including OpenAI, and TextSynth.
- Support for evaluation on adapters (e.g. LoRA) supported in Hugging Face's PEFT library.
- Support for local models and benchmarks.

The Language Model Evaluation Harness is also the backend for 🤗 [Hugging Face's (HF) popular Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Table of Contents

- [Installation](#installation)
- [How to Run Evaluation](#how-to-run-evaluation)
  - [Quick Test](#quick-test)
  - [PEFT Fine-tuned model Evaluation](#peft-fine-tuned-model-evaluation)
  - [Reproducing Hugging Face Open-LLM-Leaderboard](#reproducing-hugging-face-open-llm-leaderboard)
  - [Multi-GPU Evaluation](#multi-gpu-evaluation)
  - [Custom Task Evaluation](#custom-task-evaluation)

## Installation

Before running the evaluation script, ensure you have all the necessary dependencies installed.

```bash
cd lm-evaluation-harness
pip install -e .
```
## How to Run Evaluation

### Quick Test

To run evaluation for Hugging Face `Llama-3-8B-Instruct` model on a single GPU please run the following,

```bash
python eval.py --model "hf" --model_args pretrained=${path_to_saved_converted_hf_model} --tasks hellaswag --device cuda:0 --batch_size 8
```

Tasks can be extended by using `,` between them for example `--tasks hellaswag,arc`.

To set the number of shots you can use `--num_fewshot` to set the number for few shot evaluation.

> [!NOTE]
> To handle the error: ` WARNING  [huggingface.py:1317] Failed to get model SHA for ../llama3/Meta-Llama-3-8B-Instruct-hf at revision main. Error: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../llama3/Meta-Llama-3-8B-Instruct-hf'. Use `repo_type` argument if needed.`
> Actually, this error is not a problem, but it is a warning that the model SHA is not found. It is not necessary to fix it. But if you want to fix it, you can follow the steps below.
> We can change the source code in `[lm-evaluation-harness/lm_eval/models/huggingface.py:1312]`:
> from
```python
def get_model_sha(pretrained: str, revision: str) -> str:
    try:
        model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
        return model_info.sha
    except Exception as e:
        eval_logger.warn(
            f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
        )
        return ""
```
> to
```python
def get_model_sha(pretrained: str, revision: str) -> str:
    # Check if the path is a local path
    if Path(pretrained).exists():
        # Compute a SHA for the local directory (for example purposes, this can be a hash of the directory contents)
        # Here we'll just return an empty string or some static string because computing SHA for a directory is non-trivial
        return "local_directory"
    else:
        try:
            model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
            return model_info.sha
        except Exception as e:
            eval_logger.warn(
                f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
            )
            return ""
```
> [!NOTE]
> We can ignore the error: `fatal: not a git repository (or any parent up to mount point /root)`

### PEFT Fine-tuned model Evaluation

### Reproducing Hugging Face Open-LLM-Leaderboard

### Multi-GPU Evaluation

### Custom Task Evaluation
