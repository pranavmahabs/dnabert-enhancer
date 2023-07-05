import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

## TODO: change to GeLU activation function in the encoder layesr <3 
class GeneTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout):
        super().__init__()
        self.model_type = "TransformerBlock"
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, "gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_mask, predict=False):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        attention_scores = []
        for layer in self.transformer_encoder.layers:
            x = layer(x, x_mask)
            if predict:
                attn, _ = layer.self_attn(x, x, x, attn_mask=x_mask)
                attention_scores.append(attn)
        # output = self.transformer_encoder(x, x_mask)
        return x, attention_scores