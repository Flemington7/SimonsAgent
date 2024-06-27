# Fine-tuning

## Table of Contents

- [Overview](#overview)
  - [Weights & Biases Experiment Tracking](#weights--biases-experiment-tracking)
  - [FLOPS Counting and Pytorch Profiling](#flops-counting-and-pytorch-profiling)
- [single-GPU setup](#single-gpu-setup)
  - [How to run it?](#how-to-run-it)
  - [How to run with different datasets?](#how-to-run-with-different-datasets)
- [multi-GPU setup](#multi-gpu-setup)
  - [How to run it](#how-to-run-it)
  - [How to run with different datasets?](#how-to-run-with-different-datasets)

## Overview

This section contains instructions to fine-tune Meta Llama 3 on a

* [single-GPU setup](#single-gpu-setup)
* [multi-GPU setup](#multi-gpu-setup)

using the canonical [finetuning script](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/finetuning.py) in the llama-recipes package.

If you are new to fine-tuning techniques, check out an overview: [](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/LLM_finetuning_overview.md)

> [!TIP]
> If you want to try finetuning Meta Llama 3 with Huggingface's trainer, here is a Jupyter notebook with an [example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/finetuning/huggingface_trainer/peft_finetuning.ipynb)

> [!TIP]
> All the setting defined in [config files](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

* [Training config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/training.py) is the main config file that helps to specify the settings for our run and can be found in [configs folder](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/).

    It lets us specify the training settings for everything from `model_name` to `dataset_name`, `batch_size` and so on. Below is the list of supported settings:

    ```python
        model_name: str="PATH/to/Model"
        tokenizer_name: str=None
        enable_fsdp: bool=False
        low_cpu_fsdp: bool=False
        run_validation: bool=True
        batch_size_training: int=4
        batching_strategy: str="packing" #alternative: padding
        context_length: int=4096
        gradient_accumulation_steps: int=1
        gradient_clipping: bool = False
        gradient_clipping_threshold: float = 1.0
        num_epochs: int=3
        max_train_step: int=0
        max_eval_step: int=0
        num_workers_dataloader: int=1
        lr: float=1e-4
        weight_decay: float=0.0
        gamma: float= 0.85
        seed: int=42
        use_fp16: bool=False
        mixed_precision: bool=True
        val_batch_size: int=1
        dataset = "samsum_dataset"
        peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
        use_peft: bool=False
        from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
        output_dir: str = "PATH/to/save/PEFT/model"
        freeze_layers: bool = False
        num_freeze_layers: int = 1
        quantization: bool = False
        one_gpu: bool = False
        save_model: bool = True
        dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
        dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
        save_optimizer: bool=False # will be used if using FSDP
        use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        use_wandb: bool = False # Enable wandb for experient tracking
        save_metrics: bool = False # saves training metrics to a json file for later plotting
        flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
        flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
        use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
        profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    
    ```

* [Datasets config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py) provides the available options for datasets.

* [peft config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified. We currently support LoRA and Llama-Adapter. Please note that LoRA is the only technique which is supported in combination with FSDP.

* [FSDP config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/fsdp.py) provides FSDP settings such as:

  * `mixed_precision` boolean flag to specify using mixed precision, defatults to true.

  * `use_fp16` boolean flag to specify using FP16 for mixed precision, defatults to False. We recommond not setting this flag, and only set `mixed_precision` that will use `BF16`, this will help with speed and memory savings while avoiding challenges of scaler accuracies with `FP16`.

  * `sharding_strategy` this specifies the sharding strategy for FSDP, it can be:
  
    * `FULL_SHARD` that shards model parameters, gradients and optimizer states, results in the most memory savings.

    * `SHARD_GRAD_OP` that shards gradinets and optimizer states and keeps the parameters after the first `all_gather`. This reduces communication overhead specially if you are using slower networks more specifically beneficial on multi-node cases. This comes with the trade off of higher memory consumption.

    * `NO_SHARD` this is equivalent to DDP, does not shard model parameters, gradinets or optimizer states. It keeps the full parameter after the first `all_gather`.

    * `HYBRID_SHARD` available on PyTorch Nightlies. It does FSDP within a node and DDP between nodes. It's for multi-node cases and helpful for slower networks, given your model will fit into one node.

  * `checkpoint_type` specifies the state dict checkpoint type for saving the model. `FULL_STATE_DICT` streams state_dict of each model shard from a rank to CPU and assembels the full state_dict on CPU. `SHARDED_STATE_DICT` saves one checkpoint per rank, and enables the re-loading the model in a different world size.
  
  * `fsdp_activation_checkpointing` enables activation checkpoining for FSDP, this saves significant amount of memory with the trade off of recomputing itermediate activations during the backward pass. The saved memory can be re-invested in higher batch sizes to increase the throughput. We recommond you use this option.
  
  * `pure_bf16` it moves the  model to `BFloat16` and if `optimizer` is set to `anyprecision` then optimizer states will be kept in `BFloat16` as well. You can use this option if necessary.

### Weights & Biases Experiment Tracking

You can enable [W&B](https://wandb.ai/) experiment tracking by using `use_wandb` flag as below. You can change the project name, entity and other `wandb.init` arguments in `wandb_config`.

```bash
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model --use_wandb
```

You'll be able to access a dedicated project or run link on [wandb.ai](https://wandb.ai) and see your dashboard.

### FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.

## single-GPU setup

This recipe steps you through how to finetune a Meta Llama 3 model on the text summarization task using the grammar dataset on a single GPU.

These are the instructions for using the canonical [finetuning script](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/finetuning.py) in the llama-recipes package.

### How to run it?

Get access to a machine with one GPU (in this case we tested with 1 A100).

```bash
torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --use_fp16 --model_name ${path_to_saved_converted_hf_model} --output_dir ${path_to_save_peft_model} --dataset ${dataset} --train_split ${path_to_train_split} --test_split ${path_to_test_split}
```

The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script
* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.
* `--quantization` boolean flag to enable int8 quantization
* `--dataset` to specify the dataset to use, the default is `samsum_dataset`
* `--train_split` to specify the path to the training split
* `--test_split` to specify the path to the test split

> [!NOTE]
> In case you are using a multi-GPU machine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`.
> The --output_dir flag specifies the path to save the PEFT model, and it should not contain any '_', otherwise the PEFT script will not be able to load the model!

### How to run with default datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/datasets/README.md)).

* `grammar_dataset`: use this [notebook](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

    To run with the grammar dataset set the `dataset` flag in the command as shown below:
    
    ```bash
    # grammar_dataset
    
    torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --dataset grammar_dataset --train_split "/root/miniconda3/envs/llama3/lib/python3.11/site-packages/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" --test_split "/root/miniconda3/envs/llama3/lib/python3.11/site-packages/llama_recipes/datasets/grammar_dataset/grammar_validation.csv" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_grammar_dataset_20240522" --save_model False --use_fp16 --use_fast_kernels --one_gpu --context_length 1024 --use_wandb
    ```

### How to run with custom datasets?

In order to start a training with the custom dataset we need to set the `--dataset` as well as the `--custom_dataset.file` parameter.

```bash
torchrun --nnodes 1 -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "path/to/custom_dataset.py" --train_split ${path_to_train_split} --test_split ${path_to_test_split}
```

e.g.

```bash
# custom_dataset

torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --dataset "custom_dataset" --custom_dataset.file "../fine-tuning/custom_dataset.py" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "Meta-Llama-3-8B-Instruct-hflora-custom-dataset-2024060x" --save_model False --use_fp16 --use_fast_kernels --one_gpu --context_length 1024
```

```bash
# sentiment_dataset

torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --dataset "custom_dataset" --custom_dataset.file "../fine-tuning/custom_dataset.py" --train_split "../fine-tuning/datasets/sentiment_train.csv" --test_split "../fine-tuning/datasets/sentiment_validation.csv" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "../fine-tuning/pefe-model/Meta-Llama-3-8B-Instruct-hf-lora-sentiment-dataset-20240606" --use_fp16 --use_fast_kernels --one_gpu --context_length 1024 --num_epochs 30
```

> [!TIP]
> The custom dataset should be a python file that contains the dataset class and the data processing functions. You can find more information on how to create a custom dataset [here](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/datasets/README.md).

#### Batching Strategy (TBD)

## multi-GPU setup

This recipe steps you through how to finetune a Meta Llama 3 model on the text summarization task using the grammar dataset on multiple GPUs in a single nodes.

### How to run it
Get access to a machine with multiple GPUs (in this case we tested with 2 A100).

```bash
torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.finetuning --enable_fsdp --use_peft --peft_method lora --model_name ${path_to_saved_converted_hf_model} --output_dir ${path_to_save_peft_model} --dataset ${dataset} --train_split ${path_to_train_split} --test_split ${path_to_test_split}
```

We use `torchrun` to spawn multiple processes for FSDP.

The args used in the command above are:
* `--enable_fsdp` boolean flag to enable FSDP in the script
* `--use_peft` boolean flag to enable PEFT methods in the script
* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.
* `--dataset` to specify the dataset to use, the default is `samsum_dataset`
* `--train_split` to specify the path to the training split
* `--test_split` to specify the path to the test split

### How to run with default datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/datasets/README.md)).

* `grammar_dataset`: use this [notebook](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

    To run with the grammar dataset set the `dataset` flag in the command as shown below:
    
    ```bash
    # grammer_dataset
    
    torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.finetuning --enable_fsdp --model_name "Meta-Llama-3-8B-Instruct-hf" --use_peft --peft_method lora --dataset grammar_dataset --train_split "/root/miniconda3/envs/llama3/lib/python3.10/site-packages/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" --test_split "/root/miniconda3/envs/llama3/lib/python3.10/site-packages/llama_recipes/datasets/grammar_dataset/grammar_validation.csv" --dist_checkpoint_root_folder "model_checkpoints" --dist_checkpoint_folder "fine-tuned" --use_fast_kernels --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_grammer_dataset_20240523" --context_length 1024 --num_epochs 10
    ```

### How to run with custom datasets? (TBD)

In order to start a training with the custom dataset we need to set the `--dataset` as well as the `--custom_dataset.file` parameter.

```python
python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "path/to/custom_dataset.py"
```

e.g.

```bash
# custom_dataset(OpenAssistant/oasst1)

torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.finetuning --enable_fsdp --use_peft --peft_method lora --quantization --dataset "custom_dataset" --custom_dataset.file "../fine-tuning/custom_dataset.py" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_custom_dataset_20240605" --dist_checkpoint_root_folder "model_checkpoints" --dist_checkpoint_folder "fine-tuned" --use_fast_kernels  --context_length 1024
```

```bash
# sentiment_dataset

torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --dataset "custom_dataset" --custom_dataset.file "../fine-tuning/custom_dataset.py" --train_split "../fine-tuning/sentiment_train.csv" --test_split "../fine-tuning/sentiment_validation.csv" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_sentiment_dataset_20240605" --save_model False --pure_bf16 --use_fast_kernels --context_length 1024 --num_epochs 30
```
