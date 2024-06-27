# FinLlama: Llama3 Model Fine-tuned with Financial Data

The 'FinLLAMA' repository is a companion to the [Meta Llama 3](https://github.com/meta-llama/llama3) models. The goal of this repository is to provide examples and recipes to get started using the Llama models from Meta with financial data. The repository contains examples of how to fine-tune the Llama models on financial data, how to evaluate the models, and how to use the models for various financial tasks.

## Table of Contents

- [Llama Recipes: Examples to get started using the Llama models from Meta with financial data](#examples-to-get-started-using-the-llama-models-from-meta-with-financial-data)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
      - [Install with pip](#install-with-pip)
    - [Getting the Llama models](#getting-the-llama-models)
      - [Model conversion to Hugging Face](#model-conversion-to-hugging-face)
  - [Fine-tuning](#fine-tuning)
    - [Overview](#overview)
    - [single-GPU setup](#single-gpu-setup)
    - [multi-GPU setup](#multi-gpu-setup)
    - [Custom Datasets (TBD)](#custom-datasets-tbd)
  - [Evaluation (TBD)](#evaluation-tbd)
  - [Inference (TBD)](#inference-tbd)
    - [MP](#mp)
    - [Finetuned Models (TBD)](#fine-tuned-models-tbd)
  - [Contributing (TBD)](#contributing)
  - [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installation
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

> [!NOTE]
> Ensure you use the correct CUDA version (from `nvidia-smi`) when installing the PyTorch wheels. Here we are using 12.1 as `cu121`.

#### Install with pip

```bash
pip install -r requirements.txt
```

### Getting the Meta Llama models

In order to download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept Meta's License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

You can find more information on how to get a quick start with the native models [here](llama3/README.md).

1. In the [llama3](llama3) directory run:

    ```bash
    pip install -e .
    ```

2. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

3. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

4. Once you get the email, navigate to llama3 repository and run the download.sh script.
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

5. Once the model/s you want have been downloaded, you can run the model locally using the command below:

    ```bash
    torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 512 --max_batch_size 6
    ```

**Note**
- Replace `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory and `Meta-Llama-3-8B-Instruct/tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#mp) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](llama3/example_chat_completion.py) found in this repository but you can change that to a different .py file.

#### Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Meta Llama model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under {path_to_meta_downloaded_model} you can install all requirements and convert the checkpoint with:

```bash
conda create -n hf-convertor python=3.10
conda activate hf-convertor
pip install transformers torch tiktoken blobfile accelerate
``` 

```bash
python3 -m transformers.models.llama.convert_llama_weights_to_hf --input_dir ${path_to_meta_downloaded_model} --output_dir ${path_to_save_converted_hf_model} --model_size 8B --llama_version 3
``` 

e.g.

```bash
python3 -m transformers.models.llama.convert_llama_weights_to_hf --input_dir Meta-Llama-3-8B-Instruct --output_dir Meta-Llama-3-8B-Instruct-hf  --model_size 8B --llama_version 3
```

## Fine-tuning

### Overview

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

#### Weights & Biases Experiment Tracking

You can enable [W&B](https://wandb.ai/) experiment tracking by using `use_wandb` flag as below. You can change the project name, entity and other `wandb.init` arguments in `wandb_config`.

```bash
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model --use_wandb
```

You'll be able to access a dedicated project or run link on [wandb.ai](https://wandb.ai) and see your dashboard.

#### FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.

### single-GPU setup

This recipe steps you through how to finetune a Meta Llama 3 model on the text summarization task using the grammar dataset on a single GPU.

These are the instructions for using the canonical [finetuning script](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/finetuning.py) in the llama-recipes package.

#### How to run it?

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

#### How to run with different datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/datasets/README.md)).

* `grammar_dataset`: use this [notebook](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

    To run with the grammar dataset set the `dataset` flag in the command as shown below:
    
    ```bash
    # grammar_dataset
    
    torchrun --nnodes 1 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --dataset grammar_dataset --train_split "/root/miniconda3/envs/llama3/lib/python3.11/site-packages/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" --test_split "/root/miniconda3/envs/llama3/lib/python3.11/site-packages/llama_recipes/datasets/grammar_dataset/grammar_validation.csv" --model_name "Meta-Llama-3-8B-Instruct-hf" --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_samsum_dataset_20240522" --save_model False --use_fp16 --use_fast_kernels --one_gpu --context_length 1024 --use_wandb
    ```

* `custom_dataset`: TBD.

### multi-GPU setup

This recipe steps you through how to finetune a Meta Llama 3 model on the text summarization task using the grammar dataset on multiple GPUs in a single nodes.

#### How to run it
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

#### How to run with different datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](https://github.com/meta-llama/llama-recipes/tree/main/recipes/finetuning/datasets/README.md)).

* `grammar_dataset`: use this [notebook](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

    To run with the grammar dataset set the `dataset` flag in the command as shown below:
    
    ```bash
    # grammer_dataset
    
    torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.finetuning --enable_fsdp --model_name "Meta-Llama-3-8B-Instruct-hf" --use_peft --peft_method lora --dataset grammar_dataset --train_split "/root/miniconda3/envs/llama3/lib/python3.10/site-packages/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" --test_split "/root/miniconda3/envs/llama3/lib/python3.10/site-packages/llama_recipes/datasets/grammar_dataset/grammar_validation.csv" --dist_checkpoint_root_folder "model_checkpoints" --dist_checkpoint_folder "fine-tuned" --use_fast_kernels --output_dir "Meta-Llama-3-8B-Instruct-hf_lora_grammer_dataset_20240523" --context_length 1024 --num_epochs 10
    ```

* `custom_dataset`: TBD.

### Custom Datasets (TBD)
The provided fine tuning scripts allows you to select between three datasets by passing the `dataset` arg to the `llama_recipes.finetuning` module or [`recipes/finetuning/finetuning.py`](../finetuning.py) script. The current options are `grammar_dataset`, `alpaca_dataset`and `samsum_dataset`. Additionally, llama-recipes integrate the OpenAssistant/oasst1 dataset as an [example for a custom dataset](custom_dataset.py) 

## Evaluation (TBD)

## Inference (TBD)

### MP

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

All models support sequence length up to 8192 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Fine-tuned Models (TBD)



## Contributing (TBD)



## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
