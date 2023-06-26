import torch
import torch.nn as nn
import numpy as np
import matplotlib
from model import GenoClassifier


## TODO: Write a function test the predictive capabilities of the model. 


## TODO:Write a function to read in the accuracy scores.



## TODO:Write a function that uses extracted attention scores to create a plot
# This plot would need to overlay all four lines (from one sequence)


# TODO:Write a function that searches for motifs based on attention scores

## TODO:Load in the Model!
PATH = ""
d_hid = 200
nlayers = 3
nhead = 8
dropout = 0.2
classifier = GeneClassifier()

classifier.load_state_dict(torch.load(PATH))
classifier.eval(True)

## TODO:Load in the Pickle File Containing the Data
with open(results_dir + 'data.p', 'rb') as handle:
    dataset = pickle.load(handle)

## TODO:Load in the Test Set
test_seq, test_lab = dataset['test_seq'], dataset['test_lab']
test_mask = np.ones(test_seq.shape)
test_dataset = SequenceDataset(test_seq, test_mask, test_lab)
test_loader = DataLoader(test_dataset)


## TODO:Write a function that serves as a forward hook. 
activation = {}
def getAttention(name):
    def hook(model, input, output):
        activation[name] = output.detach
    return hook

h = classifier.transformer
