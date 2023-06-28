import sys
import pickle
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory
from preprocessing import create_pickle
from data_utils import tokenizer, SequenceDataset
from model import GenoClassifier
import tranformers


pos_bed_file = sys.argv[1]
neg_bed_file = sys.argv[2]
results_dir = "data/"
fasta = "/data/Dcode/common/hg38.fa"
K = 4
# create_pickle(pos_bed_file, neg_bed_file, results_dir, fasta, K=K, build_v=False)

# TODO: Read in the Vocab and the Pickle File
with open(results_dir + "data.p", "rb") as handle:
    dataset = pickle.load(handle)
tkr = dataset["tokenizer"]