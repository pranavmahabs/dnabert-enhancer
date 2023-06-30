import torch
import torch.nn as nn
import numpy as np
import matplotlib
from model import GenoClassifier


## TODO: Write a function test the predictive capabilities of the model. 
def predict_single():
    pass

def predict():
    pass

## TODO:Write a function that uses extracted attention scores to create a plot
# This plot would need to overlay all four lines (from one sequence)
def plot_attention_one():
    pass

# TODO:Write a function that searches for motifs based on attention scores
def motif_analysis():
    # Use DNABERT for this functionality
    pass

## Load in the Pickle File Containing the Data
with open(results_dir + 'data.p', 'rb') as handle:
    dataset = pickle.load(handle)

## TODO:Load in the Test Set
test_seq, test_lab = dataset['test_seq'], dataset['test_lab']
test_mask = np.ones(test_seq.shape)
test_dataset = SequenceDataset(test_seq, test_mask, test_lab)
test_loader = DataLoader(test_dataset)

## Set the same model parameters from Training. 
PATH = "/data/Dcode/pranav/genoscanner/data"
ntokens = len(tkr.kmer2idx.keys())
d_model = 256
d_hidden = 512
nlayers = 3
nhead = 4
dropout = 0.2
num_classes = 3

# Load in the Model and Load the Parameters from Training
model = GenoClassifier(ntokens, d_model, d_hidden, nlayers, nhead, dropout, num_classes)
model.load_state_dict(torch.load(PATH))
model.eval(True)

# Predict Values and Get Attention Scores