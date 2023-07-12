import csv
import sys
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from tokenizer import (
    DNATokenizer,
    PRETRAINED_INIT_CONFIGURATION,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    VOCAB_KMER,
)


# TODO: Reimplement to reflect the dataset for DNABERT-1
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = 6,
    ):
        assert kmer in [3, 4, 5, 6], "kmer must be 3, 4, 5, or 6"
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f, delimiter="\t"))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            print("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            print("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            # if torch.distributed.get_rank() not in [0, -1]:
            #     torch.distributed.barrier()

            print(f"Tokenizing input with {kmer}-mer as input...")
            with open(data_path, "r", newline="\n") as file:
                reader = csv.reader(file, delimiter="\t")
                texts = list(reader)

            # Drop the header - note that this will drop the first sample
            #                   if the header is not included.
            texts = texts[1:]

            texts = np.asarray(texts)
            labels = texts[:, 1]
            texts = list(texts[:, 0])
            # if torch.distributed.get_rank() == 0:
            #     torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels.astype(np.int8)
        if -1 in self.labels:
            self.labels += 1
        self.num_labels = len(set(self.labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def getweights(self):
        unique_labels = list(set(self.labels))
        counts = []
        for label in unique_labels:
            counts.append(np.count_nonzero(self.labels == label))
        top = max(counts)
        weights = [top / count for count in counts]
        return torch.Tensor(weights)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def pickle_dataset(tsv_files, config):
    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[config]["do_lower_case"],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[config],
    )
    outfile = "supervised_dataset.p"
    to_dump = {}
    for tsv_file, name in zip(tsv_files, ["val", "train", "test"]):
        dataset = SupervisedDataset(tsv_file, tokenizer)
        to_dump[name] = dataset

    with open(outfile, 'wb') as handle:
        pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    file_base = "/data/Dcode/pranav/genoscanner/data/"
    files = [file_base + "val.tsv", file_base + "train.tsv", file_base + "test.tsv"]
    config = "dna6"
    pickle_dataset(files, config)
