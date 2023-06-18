import numpy as np
import torch
import torch.nn as nn

def positional_emboding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return pos_encoding.type(torch.FloatTensor)
    # return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = torch.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x *= torch.sqrt(tf.cast(self.d_model, tf.float32))
    x *= torch.sqrt(self.d_model.type(torch.FloatTensor))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x