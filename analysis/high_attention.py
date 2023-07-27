"""
Extractions Regions of High Attention and Exports to Fasta File.
This FASTA file will be analyzed FIMO and other motif exploration tools.
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd


def kmer2seq(kmers):
    """
    Convert kmers to original sequence (source: DNABERT)

    Arguments:
    kmers -- str, kmers separated by space.

    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers  (source: DNABERT)

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer:
        sentence += original_string[i : i + kmer] + " "
        i += stride

    return sentence[:-1].strip('"')


def contiguous_regions(condition, len_thres=5):
    """
    (source: DNABERT)

    Modified from and credit to: https://stackoverflow.com/a/4495197/3751373
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Arguments:
    condition -- custom conditions to filter/select high attention
            (list of boolean arrays)

    Keyword arguments:
    len_thres -- int, specified minimum length threshold for contiguous region
        (default 5)

    Returns:
    idx -- Index of contiguous regions in sequence

    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)

    # eliminate those not satisfying length of threshold
    idx = idx[np.argwhere((idx[:, 1] - idx[:, 0]) >= len_thres).flatten()]
    return idx


def find_high_attention(score, min_len=5, **kwargs):
    """
    (source: DNABERT)

    With an array of attention scores as input, finds contiguous high attention
    sub-regions indices having length greater than min_len.

    Arguments:
    score -- numpy array of attention scores for a sequence

    Keyword arguments:
    min_len -- int, specified minimum length threshold for contiguous region
        (default 5)
    **kwargs -- other input arguments:
        cond -- custom conditions to filter/select high attention
            (list of boolean arrays)

    Returns:
    motif_regions -- indices of high attention regions in sequence

    """

    cond1 = score > np.mean(score)
    cond2 = score > 10 * np.min(score)
    cond = [cond1, cond2]

    cond = list(map(all, zip(*cond)))

    if "cond" in kwargs:  # if input custom conditions, use them
        cond = kwargs["cond"]
        if any(
            isinstance(x, list) for x in cond
        ):  # if input contains multiple conditions
            cond = list(map(all, zip(*cond)))

    cond = np.asarray(cond)

    # find important contiguous region with high attention
    motif_regions = contiguous_regions(cond, min_len)

    return motif_regions


def attention_seq(
    attentions, sequences, out_dir, min_len=5, window=4, indices=None, name="All"
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(attentions.shape) == 2:
        print("Processing Single Attention Scores.")
        with open(
            os.path.join(out_dir, "{}_high_attention.fa".format(name)), "w"
        ) as fa:
            for index, attention, sequence in zip(
                range(attentions.shape[0]), attentions, sequences
            ):
                motif_regions = find_high_attention(attention, min_len)
                for motif_idx in motif_regions:
                    safe_start = max(motif_idx[0] - window, 0)
                    safe_end = min(motif_idx[1] + window, len(sequence))
                    seq = sequence[safe_start:safe_end]
                    to_write = ">{} at Position: {} - {}\n".format(
                        index, safe_start, safe_end
                    )
                    to_write += seq + "\n"
                    fa.write(to_write)

    elif len(attentions.shape) == 3:
        print("Processing Multi-Head Attention Scores.")
        for head in range(attentions.shape[1]):
            file_name = os.path.join(
                out_dir, "{}_head{}_high_attention.fa".format(name, head)
            )
            with open(file_name, "w") as hfa:
                for index, attention, sequence in zip(
                    range(attentions.shape[0]), attentions, sequences
                ):
                    this_attention = attention[head]
                    motif_regions = find_high_attention(this_attention, min_len)
                    for motif_idx in motif_regions:
                        seq = sequence[motif_idx[0] : motif_idx[1]]
                        to_write = ">Sequence {} at Position: {} - {}\n".format(
                            index, motif_idx[0], motif_idx[1]
                        )
                        to_write += seq + "\n"
                        hfa.write(to_write)
    else:
        print("Failed. Improper attention score array provided.")
        sys.exit(1)


import matplotlib.pyplot as plt

colors = [
    "blue",  # Blue
    "green",  # Green
    "red",  # Red
    "purple",  # Purple
    "orange",  # Orange
    "brown",  # Brown
    "pink",  # Pink
    "gray",  # Gray
    "olive",  # Olive
    "cyan",  # Cyan
    "magenta",  # Magenta
    "teal",  # Teal
]


def plot_line(values, name):
    x_values = list(range(len(values)))

    # Create the plot
    plt.plot(x_values, values, marker="o", linestyle="-", color="b")

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Height")
    plt.title(f"Average Attention Scores for {name}")

    # Show the plot
    plt.show()


def plot_multi_line(list_values):
    x_values = list(range(len(list_values[0])))
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(x_values, list_values[i], linestyle="-", color=colors[i])
        ax.set_title(f"Plot for Head {i+1}")
        ax.set_xlabel("Base Pair")
        ax.set_ylabel("Average Attention for Head")

    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_random(attentions):
    random_indices = np.random.choice(len(attentions), 12, replace=False)
    sample = attentions[random_indices]

    x_values = list(range(len(attentions[0])))
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(x_values, sample[i], linestyle="-", color=colors[i])
        ax.set_title(f"Plot for Sample {i+1}")
        ax.set_xlabel("Base Pair")
        ax.set_ylabel("Attention Score")

    plt.tight_layout()

    # Show the plot
    plt.show()
