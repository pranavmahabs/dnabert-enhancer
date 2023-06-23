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

pos_bed_file = sys.argv[1]
neg_bed_file = sys.argv[2]
results_dir = "gs_results/"
fasta = sys.argv[3]
K = "4"
create_pickle(pos_bed_file, neg_bed_file, results_dir, K=K, build_v=True)

# TODO: Read in the Vocab and the Pickle File
with open('data.p', 'rb') as handle:
    dataset = pickle.load(handle)
tkr = dataset['tokenizer']

# Set up CUDA Configurations with the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create and Define Batches
# Load the Data
train_seq, train_lab = dataset['train_seq'], dataset['train_lab']
train_mask = np.ones(train_seq.shape)
train_dataset = SequenceDataset(train_seq, train_mask, train_lab)

val_seq, val_lab = dataset['val_seq'], dataset['val_lab']
val_mask = np.ones(val_seq.shape)
val_dataset = SequenceDataset(val_seq, val_mask, val_lab)

test_seq, test_lab = dataset['test_seq'], dataset['test_lab']
test_mask = np.ones(test_seq.shape)
test_dataset = SequenceDataset(test_seq, test_mask, test_lab)


# Prepare Dataset!
BATCH_SIZE = 64
GPUS = 4
EPOCH = 10
num_samples = len(train_seq)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the Model
ntokens = len(tkr.kmer2idx.keys)
d_hid = 200
nlayers = 3
nhead = 6
dropout = 0.2
model = GenoClassifier(ntokens, d_hid, nlayers, nhead, dropout)

# Train Function
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
num_batches = len(train_loader)


def train_epoch(epoch):
    model.train(True)
    total_loss = 0.
    last_loss = 0.
    log_interval = 200
    start_time = time.time()

    for i, data in enumerate(train_loader):
        # get batch and split into seqs, masks, labels
        sequences, masks, labels = data
        # Zero Gradients for every batch
        optimizer.zero_grad()
        # Make predictions for this batch
        output = model(sequences, masks)
        # Update Learning Weights
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # Print Status
        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(loader: DataLoader):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            vseqs, vmasks, vlabels = data
            voutputs = model(vseqs, vmasks)
            vloss = criterion(voutputs, vlabels)
            total_loss += vloss
    return total_loss / (len(val_loader))


# TRAINING: Epochs Loop
best_val_loss = float('inf')
# with TemporaryDirectory() as tempdir:
best_model_params_path = os.path.join(results_dir, "best_model_params.pt")

for epoch in range(1, EPOCH + 1):
    epoch_start_time = time.time()
    print('EPOCH {}:'.format(epoch + 1))
    train_epoch(epoch)
    vloss = evaluate()
    val_ppl = math.exp(vloss)
    elapsed = time.time() - epoch_start_time

    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {vloss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if vloss < best_val_loss:
        best_val_loss = vloss
        torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()
