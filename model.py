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

# model architecture
class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 nhead: int,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()
        #transformer architecture
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        
        # Positional encoding
        self.positional_encoder = PositionalEncoder(d_model) 
        #linear layer to convert output to probabilities (final stage)
        self.generator = nn.Linear(d_model, target_vocab_size)

        #create embedding layer to convert tokens to vectors
        self.source_tok_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_tok_embedding = nn.Embedding(target_vocab_size, d_model)
    
    #src are input tokens for source sequence, trg input tokens for target sequence
    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor,
                source_padding_mask: torch.Tensor,
                target_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        #convert tokens to vectors and add their positions
        source_embedding = self.positional_encoder(self.source_tok_embedding(source))
        target_embedding = self.positional_encoder(self.target_tok_embedding(target))
        #feed the source and target data to the transformer
        outs = self.transformer(source_embedding, target_embedding, source_mask, target_mask, None,
                                source_padding_mask, target_padding_mask, memory_key_padding_mask)
        #convert the transformer output to probabilities and then return
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encode(self.positional_encoder(
                            self.source_tok_embedding(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decode(self.positional_encoder(
                          self.target_tok_embedding(tgt)), memory,
                          tgt_mask)
    