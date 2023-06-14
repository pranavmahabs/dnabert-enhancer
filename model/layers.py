## There are six decoder blocks in the transformer model. 
import pytorch as torch

class Embedder(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len = 250):
        super().__init()
        self.d_model = d_model

    def forward(self, x):
        pass



# yay!