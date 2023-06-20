import torch
import torch.nn as nn
from embed import Embedder, PositionalEncoder
from transformer import TransformerBlock

class GenoScanner(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.transformer = TransformerBlock(vocab_size, d_model, heads, d_model, N, dropout)

        ## Classifier
        self.ff = self.Linear(d_model, d_ff)
        self.drop = nn.dropout(dropout)
        self.ff2 = self.Linear(d_model, 3)
    
    def forward(self, x, x_mask):
        src = self.transformer(x, x_mask)
        ff1 = self.drop(self.ff(src))
        logits = nn.Softmax(self.ff2(ff1))
        return logits