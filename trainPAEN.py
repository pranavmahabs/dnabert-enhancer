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

### Ask Di why she chose to use these chromosomes in the split - were they optimal? Should we do K-Fold validation?

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
FASTA_FILE = "/data/Dcode/common/hg38.fa"
WORK_DIR = "/home/mahableshwarkps/PoisedEnModel/"


def get_chrom2seq(FASTA_FILE, capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq

def seq2one_hot(seq):

    d = np.array(['A', 'C', 'G', 'T'])

    return np.fromstring(str(seq.upper()), dtype='|S1')[:, np.newaxis] == d
    
def train_model(X_train,Y_train, X_val,Y_val,X_test, Y_test, save_dir):
    

    model_file = WORK_DIR + "/src/model.hdf5"
    model = load_model(model_file)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    weights_file = os.path.join(save_dir, "model_weights.hdf5")
    from keras.utils.np_utils import to_categorical
    from sklearn.utils import class_weight
    
    print(Y_train.mean(axis=0), Y_train.shape)
    print(Y_test.mean(axis=0), Y_test.shape)   
    print(Y_val.mean(axis=0), Y_val.shape)
    
    # Change the label space from -1, 0, 1 to 0 (PE), 1 (Noise), 2 (AE)
    Y_train = Y_train+1
    Y_test = Y_test+1
    Y_val = Y_val+1

    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    print(class_weights)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print(model.summary())

    _callbacks = []
    checkpoint = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    _callbacks.append(checkpoint)
    earlystopping = EarlyStopping(monitor="val_loss", patience=15)
    _callbacks.append(earlystopping)

    model.fit(X_train, Y_train,
                       batch_size=500, 
                       epochs=15,
                       validation_data=(X_val, Y_val), class_weight = class_weights,
                       shuffle=True,
                       callbacks=_callbacks, verbose=1)

    model.load_weights(weights_file) 
    Y_pred = model.predict(X_test)
    
    test_result_file = os.path.join(save_dir, "testresult.hdf5")
    with h5py.File(test_result_file, "w") as of:
        of.create_dataset(name="pred", data=Y_pred, compression="gzip")
        of.create_dataset(name="label", data=Y_test, compression="gzip")

    # Dictates the AUC Score - False Positive Rate. Closer to 1 is indicative of a better model. 
    auc1 = metrics.roc_auc_score((Y_test==0)+0, Y_pred[:,0])
    auc2 = metrics.roc_auc_score((Y_test==2)+0, Y_pred[:,2])

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("Active Enhancer AUC: %f\n" % auc2)
        of.write("Poised Enhancer AUC: %f\n" % auc1) 

    [fprs, tprs, thrs] = metrics.roc_curve((Y_test==0)+0, Y_pred[:, 0])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thrs[sort_ix[0]]

    [fprs, tprs, thrs] = metrics.roc_curve((Y_test==2)+0, Y_pred[:, 2])
    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thre = thrs[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thre = thrs[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("sl10 \t %f\n" % fpr10_thr)
        of.write("5 \t %f\n" % fpr5_thr)
        of.write("3 \t %f\n" % fpr3_thr)
        of.write("1 \t %f\n" % fpr1_thr)
       	of.write("en10 \t %f\n" % fpr10_thre)
        of.write("5 \t %f\n" % fpr5_thre)
        of.write("3 \t %f\n" % fpr3_thre)
        of.write("1 \t %f\n" % fpr1_thre)


def get_chrom2seq(capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq


if __name__ == "__main__":

    import sys
    posbed = sys.argv[1]
    negbed = sys.argv[2]
    results_dir = "poisedEnhancer.CNNResults"
    create_dataset(posbed,negbed,results_dir)
