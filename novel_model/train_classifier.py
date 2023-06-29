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
results_dir = "data/"
fasta = "/data/Dcode/common/hg38.fa"
K = 4
# create_pickle(pos_bed_file, neg_bed_file, results_dir, fasta, K=K, build_v=False)

# TODO: Read in the Vocab and the Pickle File
with open(results_dir + "data.p", "rb") as handle:
    dataset = pickle.load(handle)
tkr = dataset["tokenizer"]

#####################################################################
#                      Configuring Model/GPUs                       #
#####################################################################

# Instantiate the Model
ntokens = len(tkr.kmer2idx.keys())
d_model = 256
d_hidden = 512
nlayers = 3
nhead = 4
dropout = 0.2
num_classes = 3
model = GenoClassifier(ntokens, d_model, d_hidden, nlayers, nhead, dropout, num_classes)

# Set up CUDA Configurations with the GPU
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), torch.cuda.get_device_name(0))
print(device)
model = nn.DataParallel(model)
model.to(device)
use_amp = True

pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params, trainable)


#####################################################################
#                        Loading the Dataset                        #
#####################################################################

# Create and Define Batches
# Load the Data
train_seq, train_lab = dataset["train_seq"], dataset["train_lab"]
train_mask = np.ones(train_seq.shape)
train_dataset = SequenceDataset(train_seq, train_mask, train_lab, device)
print(train_seq.shape, train_lab.shape, train_mask.shape)

val_seq, val_lab = dataset["val_seq"], dataset["val_lab"]
val_mask = np.ones(val_seq.shape)
val_dataset = SequenceDataset(val_seq, val_mask, val_lab, device)

test_seq, test_lab = dataset["test_seq"], dataset["test_lab"]
test_mask = np.ones(test_seq.shape)
test_dataset = SequenceDataset(test_seq, test_mask, test_lab, device)

# Prepare Dataset!
BATCH_SIZE = 128
# GPUS = 4
EPOCH = 5
num_samples = len(train_seq)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


#####################################################################
#                      Model Training/Validation                    #
#####################################################################


# Train Function
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
num_batches = len(train_loader)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train_epoch(epoch):
    model.train(True)
    total_loss = 0.0
    last_loss = 0.0
    log_interval = 200
    start_time = time.time()

    for i, data in enumerate(train_loader):
        # get batch and split into seqs, masks, labels
        sequences = data["Sequence"]
        masks = torch.ones(sequences.size(dim=0), sequences.size(dim=0))
        labels = data["Class"]
        optimizer.zero_grad()

        with torch.autocast(device_type=device_name, dtype=torch.float16):
            # Make predictions for this batch
            output, attn = model(sequences, device)  # removed masks
            assert output.dtype is torch.float16
            # Update Learning Weights
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        # Print Status
        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | "
                f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
            )
            total_loss = 0
            start_time = time.time()


def evaluate(loader: DataLoader):  # validation
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            vseqs, vlabels = data["Sequence"], data["Class"]
            voutputs, vattn = model(vseqs, device)  # removed masks
            vloss = criterion(voutputs, vlabels)
            total_loss += vloss

            predicted_labels = torch.argmax(voutputs, dim=1)
            true_labels = torch.argmax(vlabels, dim=1)
            total_correct += (predicted_labels == true_labels).sum().item()
            total_samples += vlabels.size(dim=0)

    accuracy = total_correct / total_samples
    average_loss = total_loss / len(loader)
    return accuracy, average_loss


# TRAINING: Epochs Loop
best_val_loss = float("inf")
# with TemporaryDirectory() as tempdir:
best_model_params_path = os.path.join(results_dir, "best_model_params.pt")

for epoch in range(1, EPOCH + 1):
    epoch_start_time = time.time()
    print("EPOCH {}:".format(epoch))
    train_epoch(epoch)
    accuracy, vloss = evaluate(val_loader)
    val_ppl = math.exp(vloss)
    elapsed = time.time() - epoch_start_time

    print("-" * 89)
    print(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss {vloss:5.2f} | valid ppl {val_ppl:8.2f}"
        f"valid accuracy {accuracy*100:.2f}%"
    )
    print("-" * 89)

    if vloss < best_val_loss:
        best_val_loss = vloss
        torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()
