import os
import csv
import copy
import json
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional

import torch
import pickle
from pynvml import *
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import PeftConfig, PeftModel

from data_dnabert import SupervisedDataset, DataCollatorForSupervisedDataset
from tokenizer import (
    DNATokenizer,
    PRETRAINED_INIT_CONFIGURATION,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    VOCAB_KMER,
)


@dataclass
class ModelArguments:
    model_config: str = field(
        default="dna6", metadata={"help": "Choose dna3, dna4, dna5, or dna6"}
    )
    dnabert_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Dir that has Pretrained DNABERT."},
    )
    peft_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Dir that has Finetuned PEFT DNABERT."},
    )
    label_json: str = field(
        default=None, metadata={"help": "Json with Label2Id config."}
    )
    out_dir: Optional[str] = field(
        default="evaluation_output",
        metadata={"help": "Where you want results to be saved."},
    )


@dataclass
class DataArguments:
    kmer: int = field(
        default=-1,
        metadata={"help": "k-mer for input sequence. Must be 3, 4, 5, or 6."},
    )
    data_pickle: str = field(
        default=None, metadata={"help": "ONLY accepts the pickle file from training."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    per_device_eval_batch_size: int = field(default=1)
    seed: int = field(default=42)


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "AUC_score_0":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 0), logits[:, 0]),
        "AUC_score_2":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 2), logits[:, 2]),
    }


def process_scores(attention_scores, kmer):
    softmax = torch.nn.Softmax(dim=1)
    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

    # attention_scores: (batch_size, num_heads, seq_len, seq_len)
    for index, attention_score in enumerate(attention_scores):
        # (1, num_heads, seq_len, seq_len)
        attn_score = []
        for i in range(1, attention_score.shape[-1] - kmer + 2):
            # sum (heads, 0, all_scores) -> 0: Beginning of Sentence Token
            attn_score.append(float(attention_score[:, 0, i].sum()))

        for i in range(len(attn_score) - 1):
            if attn_score[i + 1] == 0:
                attn_score[i] = 0
                break

        # attn_score[0] = 0
        counts = np.zeros([len(attn_score) + kmer - 1])
        real_scores = np.zeros([len(attn_score) + kmer - 1])
        for i, score in enumerate(attn_score):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score
        real_scores = real_scores / counts
        real_scores = real_scores / np.linalg.norm(real_scores)

        scores[index] = real_scores
    return scores


def process_multi_score(attention_scores, kmer):
    scores = np.zeros(
        [
            attention_scores.shape[0],
            attention_scores.shape[1],
            attention_scores.shape[-1],
        ]
    )

    # attention_scores: (batch_size, num_heads, seq_len, seq_len)
    for index, attention_score in enumerate(attention_scores):
        head_scores = np.zeros([attention_scores.shape[1], attention_scores.shape[-1]])
        for head in range(0, len(attention_score)):
            attn_score = []

            for i in range(1, attention_score.shape[-1] - kmer + 2):
                attn_score.append(float(attention_score[head, 0, i]))

            for i in range(len(attn_score) - 1):
                if attn_score[i + 1] == 0:
                    attn_score[i] = 0
                    break

            counts = np.zeros([len(attn_score) + kmer - 1])
            real_scores = np.zeros([len(attn_score) + kmer - 1])

            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i + j] += 1.0
                    real_scores[i + j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)

            head_scores[head] = real_scores

        scores[index] = head_scores
    return scores


def evaluate():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_args.model_config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[model_args.model_config][
            "do_lower_case"
        ],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[model_args.model_config],
    )

    print(f"Provided Pickle File: {data_args.data_pickle}")

    print("Loading the Pickled Dataset")
    with open(data_args.data_pickle, "rb") as handle:
        dataset = pickle.load(handle)

    complete_dataset = dataset["positive"]

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    with open(model_args.label_json, "r") as jfile:
        data = json.load(jfile)

    label2id = data.get("label2id", {})
    metadata = data.get("metadata", {})

    num_labels = metadata["num_labels"]
    id2label = {v: k for k, v in label2id.items()}

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.dnabert_path,
        cache_dir=None,
        num_labels=num_labels,
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id,
        output_attentions=True,
    )

    config = PeftConfig.from_pretrained(model_args.peft_path)
    inference_model = PeftModel.from_pretrained(model, model_args.peft_path)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=inference_model,
        args=train_args,
        do_train=False,
        do_predict=True,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataloader_drop_last=False,
    )

    eval_results = trainer.predict(test_dataset=complete_dataset)
    eval_attens = (eval_results.attentions[-1]).numpy()
    atten_scores = process_scores(eval_attens, data_args.kmer)
    eval_logits = eval_results.logits.detach().numpy()
    all_scores = process_multi_score(eval_attens, data_args.kmer)

    np.save(os.path.join(train_args.output_dir, "atten.npy"), atten_scores)
    np.save(os.path.join(train_args.output_dir, "pred_results.npy"), eval_logits)
    np.save(os.path.join(train_args.output_dir, "heads_atten.npy"), all_scores)

    os.makedirs(train_args.output_dir, exist_ok=True)
    with open(os.path.join(train_args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    evaluate()
