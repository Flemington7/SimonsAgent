# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, csv_name=None):
        try:
            # load dataset and clear cache
            self.dataset = load_dataset('csv', data_files = csv_name, download_mode="force_redownload")

            # map label to integer
            label_map = {"neutral": 0, "negative": -1, "positive": 1}
            self.dataset = self.dataset.map(lambda x: {"label": label_map[x["label"]], "text": x["text"]})
        except Exception as e:
            print("Loading of custom dataset failed!")
            raise e

        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        input_ids = self.tokenizer.encode(self.tokenizer.bos_token + example_batch["text"] + self.tokenizer.eos_token,
                                          add_special_tokens=False)

        sample = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "label": example_batch["label"]
        }

        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset["train"][int(index)])

def get_custom_dataset(dataset_config, tokenizer, csv_name=None):
    dataset = CustomDataset(
        tokenizer=tokenizer,
        csv_name=csv_name,
    )

    return dataset
