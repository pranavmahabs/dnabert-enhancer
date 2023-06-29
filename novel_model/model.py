import torch
import torch.nn as nn
from transformer import GeneTransformer


class GenoClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, d_hidden, N, heads, dropout, num_classes):
        super().__init__()
        self.transformer = GeneTransformer(
            vocab_size, d_model, heads, d_hidden, N, dropout
        )
        ## GRU Handle Sequential Output of Transformer
        LSTM_layers = 4
        self.lstm = nn.LSTM(d_model, d_hidden, LSTM_layers, batch_first=True)
        self.lstm.flatten_parameters()
        ## Classifier
        self.ff = nn.Linear(d_hidden, num_classes)

    def forward(self, x, device, predict=False):
        x_mask = torch.ones(x.size(dim=0), x.size(dim=0)).to(device)
        x_mask = x_mask.to(device)
        src, attention_scores = self.transformer(x, x_mask, predict=predict)
        lstm_output, _ = self.lstm(src)
        last_output = lstm_output[:,-1,:]
        logits = self.ff(last_output)
        return logits, attention_scores
