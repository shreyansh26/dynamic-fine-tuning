from dataclasses import dataclass
import os
from typing import Dict, Sequence
from torch.utils.data import Dataset
from trl.data_utils import pack_dataset
import datasets
import logging
import torch.distributed as dist
import torch
import torch.nn.functional as F
import transformers
import copy
import math

IGNORE_INDEX = -100

def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess_instruct(
    train_on_inputs: bool,
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = [f"{question}" for question in samples["instruction"]]
    targets = [f"{answer}" for answer in samples["output"]]
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if not train_on_inputs:
            label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

def _filter_tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    samples = []
    for text in strings:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)

    return samples

def filter_long_samples(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    sources = [f"{question}" for question in samples["instruction"]]
    targets = [f"{answer}" for answer in samples["output"]]
    examples = [s + t for s, t in zip(sources, targets)]

    return _filter_tokenize_fn(examples, tokenizer)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        train_on_inputs: bool,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths: list[str],
        limit=None,
        workers=None,
        packing=None,
    ):
        super(SupervisedDataset, self).__init__()
        if workers is None:
            workers = math.ceil(os.cpu_count() / dist.get_world_size())
        logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")
        preprocess_fn = preprocess_instruct
        dataset = (
            datasets.load_dataset(
                "json",
                data_files=data_paths,
                split=f"train[0:{limit}]" if limit else "train",
            )
            .filter(
                lambda samples: filter_long_samples(samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
            .map(
                lambda samples: preprocess_fn(train_on_inputs, samples, tokenizer),
                batched=True,
                batch_size=3000,
                num_proc=workers,
            )
        )

        if packing == "ffd":
            # Select only the tokenized columns before packing
            dataset = dataset.select_columns(["input_ids", "labels"])
            dataset = pack_dataset(dataset, tokenizer.model_max_length, "bfd")

        self.input_ids = dataset["input_ids"]
        self.labels = dataset["labels"]
        # Include position_ids from packed dataset for correct boundary detection
        if "position_ids" in dataset.column_names:
            self.position_ids = dataset["position_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )
        if hasattr(self, "position_ids"):
            item["position_ids"] = torch.tensor(self.position_ids[i])
        return item


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: int | None = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # Optionally pad to a multiple (e.g., 8) for kernel alignment or parallelism constraints
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            seq_len = input_ids.size(1)
            remainder = seq_len % self.pad_to_multiple_of
            if remainder != 0:
                pad_amount = self.pad_to_multiple_of - remainder
                input_ids = F.pad(input_ids, (0, pad_amount), value=self.tokenizer.pad_token_id)
                labels = F.pad(labels, (0, pad_amount), value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )