import sys
import pickle
import torch
import torch.nn as nn
from preprocessing import create_pickle
from data_utils import tokenizer
from model import GenoClassifier

## TODO: Read in and Load the Dataset
pos_bed_file = sys.argv[1]
neg_bed_file = sys.argv[2]
results_dir = "gs_results/"
fasta = sys.argv[3]
K="4"
create_pickle(pos, neg, data_folder, K=K, build_v=True)

## TODO: Read in the Vocab and the Pickle File
with open('data.p', 'rb') as handle:
    dataset = pickle.load(handle)
tkr = dataset['tokenizer']

## Set up CUDA Configurations with the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Create and Define Batches


## Instantiate the Model
ntokens = len(tkr.kmer2idx.keys)
d_hid = 200
nlayers = 3
nhead = 6
dropout = 0.2
model = GenoClassifier(ntokens, d_hid, nlayers, nhead, dropout)

## TODO: Train Function
import time
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module):
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // ??? ## TODO: figure this out

    for batch, i in enumerate(range(train_data.size(0) - 1), ???):
        ## TODO: get batch

        output = model(data)

        optimizer.zero_grad()
        loss.backward()
        

## TODO: Evaluate

## TODO: Epochs Loop
