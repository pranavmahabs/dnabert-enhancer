from data_utils import build_vocab, tokenizer
import os
import numpy as np
import random 
import time 
import glob 
from Bio import SeqIO
from pybedtools import BedTool

INPUT_LENGTH = 1000
K=4
tkr = tokenizer(k=4, "vocab")

train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", 
                     "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", 
                     "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
valid_chromosomes = ["chr7"]
test_chromosomes  = ["chr8", "chr9"]

def get_chrom2seq(FASTA_FILE, capitalize=TRUE):
    """
    Load in the genome fasta file to extract sequences from the BED Files. 
    """
    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq
    return chrom2seq

def create_dataset(pos_bed_file, neg_bed_file, results_dir, pad=True):

    # Load in the genome prior to building the dataset.
    chrom2seq = get_chrom2seq()
    
    print "Generating the positive dataset..."
    pos_beds = list(BedTool(pos_bed_file))

    # Add all binding sites in BED files to train/test/val based on chromosome.
    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in validation_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]
    
    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    pos_train_label = []
    pos_val_label = []
    pos_test_label = []

    for bed_list, data_list, label_list in zip([pos_train_bed, pos_val_bed, pos_test_bed],
                                               [pos_train_data, pos_val_data, pos_test_data],
                                               [pos_train_label, pos_val_label, pos_test_label]):

        for r in bed_list:
            seqs = []
            for offset in range(K):
                seqs.append(chrom2seq[r.chrom][r.start+offset:r.stop+offset])
            
            assert(len(_seq1) == len(_seq2) == len(_seq3) == len(_seq4))
            if not len(_seq1) == INPUT_LENGTH:
                continue; 
            tokenized = [tkr.ktokenize(_seq, 4) for _seq in seqs]
            if pad:
                tokenized = [tkr.pad(tokenized_seq, window_size=250) for tokenized_seq in tokenized]
            vectors = [tkr.toindex(tokenized_seq) for tokenized_seq in tokenized]
            for vectorized in vectors:
                data_list.append(vectorized)

            ## Insert the Label into the Approriate List
            ## 1:AE - Enhancers with H3K27AC; -1:PE, Enhancers only with H3K4me1
            _k = int(r[3]) - int(r[4])
            if _k == 0:
		        _k = 1
            for _ in range(K):
                label_list.append(_k)

    print len(pos_train_data)
    print len(pos_val_data)
    print len(pos_test_data)

    print "Generating the negative dataset..."

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in validation_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip([neg_train_bed, neg_val_bed, neg_test_bed],
                                   [neg_train_data, neg_val_data, neg_test_data]):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == INPUT_LENGTH:
                continue
            tokenized = tkr.ktokenize(_seq, 4)
            if pad:
                tokenized = tkr.pad(tokenized, window_size=250)
            vector = tkr.toindex(tokenized)
            data_list.append(vector)

    neg_train_label = [0 for i in range(len(neg_train_data))]
    neg_val_label = [0 for i in range(len(neg_val_data))]
    neg_test_label = [0 for i in range(len(neg_test_data))]

    print len(neg_train_data)
    print len(neg_val_data)
    print len(neg_test_data)

    # Concatenate the positive and negative datasets in preparation for final return
    train_data = pos_train_data + neg_train_data
    train_label = pos_train_label + neg_train_label
    val_data = pos_val_data + neg_val_data
    val_label = pos_val_label + neg_val_label
    test_data = pos_test_data + neg_test_data
    test_label = pos_test_label + neg_test_label

    # Return a dictionary containing all of the data! Could save as a pickle file for ease.
    return dict(
        train_seq = np.array(train_data),
        train_lab = np.array(train_label),
        val_seq = np.array(val_data),
        val_lab = np.array(val_label),
        test_seq = np.array(test_data),
        test_lab = np.array(test_label),
        kmer2idx = tkr.kmer2idx,
        idx2kmer = tkr.idx2kmer
    )

def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(create_dataset(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

if __name__ == '__main__':
    create_pickle(data)