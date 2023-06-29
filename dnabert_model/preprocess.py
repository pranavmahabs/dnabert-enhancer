import pandas as pd
import numpy as np
import sys
from Bio import SeqIO
from pybedtools import BedTool
from data_utils import seq2kmer

train_chromosomes = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
]
valid_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]


def get_chrom2seq(FASTA_FILE, capitalize=True):
    """
    Load in the genome fasta file to extract sequences from the BED Files.
    """
    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = (
            seq.seq.upper() if capitalize else seq.seq
        )
    return chrom2seq


def generate_tsv(sequences, labels, tsv_filename):
    if len(sequences) != len(labels):
        print(len(sequences), len(labels))
        print("Error. Sequences and labels are not of the same length.")

    tsv_content = ""
    for sequence, label in zip(sequences, labels):
        tsv_content += f"{sequence}\t{label}\n"

    with open(tsv_filename, 'w') as tsv_file:
        tsv_file.write(tsv_content)


def create_dataset(
    pos_bed_file,
    neg_bed_file,
    results_dir,
    fasta,
    K,
):
    """
    Generate the kmer-ized dataset in preparation of model training. 
    """

    # Load in the genome prior to building the dataset.
    chrom2seq = get_chrom2seq(fasta)

    print("Generating the CSV Dataset File")
    pos_beds = list(BedTool(pos_bed_file))
    neg_beds = list(BedTool(neg_bed_file))

    # Add all binding sites in BED files to train/test/val based on chromosome.
    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in valid_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    pos_train_label = []
    pos_val_label = []
    pos_test_label = []

    for bed_list, data_list, label_list in zip(
        [pos_train_bed, pos_val_bed, pos_test_bed],
        [pos_train_data, pos_val_data, pos_test_data],
        [pos_train_label, pos_val_label, pos_test_label],
    ):
        for r in bed_list:
            # Maximum sequence length of 512
            _seq = chrom2seq[r.chrom][r.start + 250: r.stop - 251]
            kmerized = seq2kmer(str(_seq), 6)
            data_list.append(kmerized)
            # Insert the Label into the Approriate List
            # 1:AE - Enhancers with H3K27AC; -1:PE, Enhancers only with H3K4me1
            _k = int(r[3]) - int(r[4])
            if _k == 0:
                _k = 1
            label_list.append(_k)

    print(len(pos_train_data))
    print(len(pos_val_data))
    print(len(pos_test_data))

    print("Generating the negative dataset...")

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in valid_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip(
        [neg_train_bed, neg_val_bed, neg_test_bed],
        [neg_train_data, neg_val_data, neg_test_data],
    ):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start + 250: r.stop - 251]
            kmerized = seq2kmer(str(_seq), 6)
            data_list.append(kmerized)

    neg_train_label = [0 for i in range(len(neg_train_data))]
    neg_val_label = [0 for i in range(len(neg_val_data))]
    neg_test_label = [0 for i in range(len(neg_test_data))]

    print(len(neg_train_data))
    print(len(neg_val_data))
    print(len(neg_test_data))

    # Concatenate the positive and negative datasets in preparation for final return
    train_data = pos_train_data + neg_train_data
    train_label = pos_train_label + neg_train_label
    val_data = pos_val_data + neg_val_data
    val_label = pos_val_label + neg_val_label
    test_data = pos_test_data + neg_test_data
    test_label = pos_test_label + neg_test_label

    for data, label, filename in zip(
        [train_data, val_data, test_data],
        [train_label, val_label, test_label],
        ["train.tsv", "val.tsv", "test.csv"]
    ):
        generate_tsv(data, label, filename)


if __name__ == "__main__":
    pos = sys.argv[1]
    neg = sys.argv[2]
    res = ""
    fasta = "/data/Dcode/common/genomes/hg38/hg38.fa"
    K = 6

    create_dataset(pos, neg, res, fasta, K)
