import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        #initialize a matrix of positions (d_model is size of token, with max_len being the max amount of tokens)
        positional_matrix = torch.zeros(max_len, d_model)
        #create positions in the sequence (0 to 4999)
        position = torch.arange(0., max_len).unsqueeze(1)

        #compute scaling factor and normalize
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        #use a sin function for even terms
        positional_matrix[:, 0::2] = torch.sin(position * div_term)
        #use a cosine function for odd terms
        positional_matrix[:, 1::2] = torch.cos(position * div_term)
        positional_matrix = positional_matrix.unsqueeze(0)
        self.register_buffer('positional_matrix', positional_matrix)

    def forward(self, x):
        return x + self.positional_matrix[:, :x.size(1)]
    