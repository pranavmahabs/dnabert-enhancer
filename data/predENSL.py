import os
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from sklearn import metrics
import h5py

train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", "chr11", "chr12", "chr13",
                     "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
validation_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]

BIN_LENGTH = 200
INPUT_LENGTH = 1000
EPOCH = 200
BATCH_SIZE = 64
GPUS = 4

nucleotides = ['A', 'C', 'G', 'T']
FASTA_FILE = "/data/huangd2/SanjarModel/hg38.fa"
WORK_DIR = "/home/mahableshwarkps/PoisedEnModel/"

def seq2one_hot(seq):

    d = np.array(['A', 'C', 'G', 'T'])

    return np.fromstring(str(seq.upper()), dtype='|S1')[:, np.newaxis] == d

def get_chrom2seq(capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq

def test_model(model, seq_file, chrom2seq=None):

    if not chrom2seq:
        chrom2seq = get_chrom2seq()
        # return chrom2seq

    print(seq_file)
    pos_beds = list(BedTool(seq_file))
    pos_data = []
    posid = []
    i = 0
    for r in pos_beds:
        _seq = chrom2seq[r.chrom][r.start:r.stop]
        if not len(_seq) == 1000:
            continue
        _vector = seq2one_hot(_seq)
        pos_data.append(_vector)
    	posid.append(i) 
        i = i+1

    posid = np.array(posid)
    pos_data_matrix = np.zeros((len(pos_data), INPUT_LENGTH, 4))
    for i in range(len(pos_data)):
            pos_data_matrix[i, :, :] = pos_data[i]
    print(pos_data_matrix.shape)
    pred = model.predict(pos_data_matrix)
 
    with h5py.File(seq_file+".prediction.hdf5", "w") as of:
        of.create_dataset(name="ypred", data=pred, compression="gzip")
	of.create_dataset(name="yid", data=posid, compression="gzip")

if __name__ == "__main__":

    import sys
    model_file = sys.argv[1]
    bed_file = sys.argv[2]
    
    if not os.path.exists(model_file):
        print("no dir "+model_file)
        exit()
    init_model_file = os.path.join(WORK_DIR, "src/model.hdf5")
    testmodel = load_model(init_model_file)
    testmodel.load_weights(model_file)

    test_model(testmodel,bed_file)
