import torch
import torch.nn as nn
from transformer import GeneTransformer


class GenoClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.transformer = GeneTransformer(
            vocab_size, d_model, heads, d_model, N, dropout
        )
        ## Classifier
        self.ff = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ff2 = nn.Linear(d_model, 3)

    def forward(self, x, x_mask):
        src = self.transformer(x, x_mask)
        ff1 = self.drop(self.ff(src))
        logits = self.ff2(ff1)
        return logits


class GenoScanner(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.transformer = GeneTransformer(
            vocab_size, d_model, heads, d_model, N, dropout
        )
        ## TODO: Support MLM and NSP

    def forward(self, x, x_mask):
        src = self.transformer(x, x_mask)
        ff1 = self.drop(self.ff(src))
        logits = nn.Softmax(self.ff2(ff1))
        return logits
