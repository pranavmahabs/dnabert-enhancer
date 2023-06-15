from layers import EncoderLayer
from embed import Embedder, PositionalEncoder

class GenoScanner(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = n
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, droupout), N)
        self.norm = Norm(d_model)

        ## Classifier
        self.ff = self.Linear(d_model, d_ff)
        self.drop = nn.dropout(dropout)
        self.ff2 = self.Linear(d_model, 3)
    
    def forward(self, x):
        pass

