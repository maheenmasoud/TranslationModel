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
    

#encoder architecture
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        #multihead attention layer
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        #feed forward layers
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        #normalize layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, padding_mask=None):
        # multihead attention
        attn_output, _ = self.attention(src, src, src, key_padding_mask=padding_mask)
        #add and normalize
        src = self.layer_norm1(src + attn_output)
        # feed forward
        ffn_output = self.ffn(src)
        src = self.layer_norm2(src + ffn_output)

        return self.dropout(src)

#decoder architecture
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        #first attention layer
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        #second attention layer combined with encoder output
        self.encoder_decoder_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        #feed forward layers
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        #normalization layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, src_padding_mask=None, trg_padding_mask=None, trg_mask=None):
        # multihead attention with mask
        attn_output, _ = self.attention(trg, trg, trg, key_padding_mask=trg_padding_mask, attn_mask=trg_mask)
        # add and normalize
        trg = self.layer_norm1(trg + attn_output)

        # multihead attention with encoder
        attn_output, _ = self.encoder_decoder_attention(trg, src, src, key_padding_mask=src_padding_mask)
        #add and normalize
        trg = self.layer_norm2(trg + self.dropout(attn_output))

        # feed forward then normalize
        ffn_output = self.ffn(trg)
        trg = self.layer_norm3(trg + ffn_output)

        return self.dropout(trg)

# model architecture
class Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        super(Transformer, self).__init__()
        
        #positional encoder layer
        self.pos_encoder = PositionalEncoder(d_model)

        #set device
        self.device = device

        # encoder and decoder layers
        self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward, dropout)
        
        #linear layer to convert output and softmax to make them probabilities
        self.output = nn.Linear(d_model, target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        #create embedding layer to convert tokens to vectors
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
    
    #src are input tokens for source sequence, trg input tokens for target sequence
    def forward(self, src, trg, src_padding_mask=None, trg_padding_mask=None, trg_mask=None):
        # source embedding and then scale by embedding dimension
        src = self.source_embedding(src) * math.sqrt(self.source_embedding.embedding_dim)
        #postional encoding
        src = self.pos_encoder(src)

        src = src.permute(1, 0, 2)
        #encoder
        src = self.encoder(src, src_padding_mask)

        # target embedding and then scale by embedding dimension
        trg = self.target_embedding(trg) * math.sqrt(self.target_embedding.embedding_dim)
        #positional encoding
        trg = self.pos_encoder(trg)

        trg = trg.permute(1, 0, 2)
        #decoder
        trg = self.decoder(trg, src, src_padding_mask, trg_padding_mask, trg_mask)

        # Final projection to vocab size
        output = self.softmax(self.output(trg))
        return output
    
    def generate(self, src, src_padding_mask, fre_tokenizer, max_len=50):
        """
        Generates translations for the given source sentences using greedy decoding.
        
        Args:
            src (torch.Tensor): Source sentences tensor [batch_size, src_len].
            src_padding_mask (torch.Tensor): Source padding mask [batch_size, src_len].
            fre_tokenizer (spm.SentencePieceProcessor): French tokenizer.
            max_len (int): Maximum length of the generated sentence.
        
        Returns:
            list of lists: Generated token IDs for each sentence in the batch.
        """
        batch_size = src.size(0)
        trg = torch.full((batch_size, 1), 0, dtype=torch.long).to(src.device)  # [batch_size, 1] <sos>:1 token

        for _ in range(max_len):
            trg_mask = self.create_trg_mask(trg.size(1)).to(src.device)  # [trg_len, trg_len]
            
            # Forward pass
            output = self.forward(src, trg, src_padding_mask, 
                                  trg_padding_mask=(trg == 2), 
                                  trg_mask=trg_mask)  # [trg_len, batch_size, vocab_size]
            
            # Get the last token's probabilities
            next_token_logits = output[-1, :, :]  # [batch_size, vocab_size]
            # Greedy decoding: select the token with highest probability
            next_tokens = torch.argmax(next_token_logits, dim=1).unsqueeze(1)  # [batch_size, 1]
            
            # Append the predicted tokens to the target sequence
            trg = torch.cat((trg, next_tokens), dim=1)  # [batch_size, trg_len + 1]
            
            # Check if all sentences have generated <eos>
            if (next_tokens == 1).all(): # 1 is the <eos> token
                break

        return trg.tolist()  # List of lists containing token IDs
    
    def create_trg_mask(self, size):
        """
        Generates a square subsequent mask for the target sequence.
        """
        mask = torch.tril(torch.ones((size, size), device=self.device) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
