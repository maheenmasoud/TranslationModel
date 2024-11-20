from model import Transformer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

#create a padding mask for the sequence
def create_padding_mask(seq, pad_token=0):
    return (seq == pad_token)

#create a mask for the first multihead attention layer in the decoder
def create_target_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)

class TransformerModel(pl.LightningModule):
    def __init__(self, 
                 model: Transformer,
                learning_rate: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, trg):
        #create the padding and target masks to give to model
        src_padding_mask = create_padding_mask(src, pad_token=self.pad_token)
        trg_padding_mask = create_padding_mask(trg, pad_token=self.pad_token)
        trg_target_mask = create_target_mask(trg.size(1))
        #run the transformer model with the data
        return self.model(src, trg, trg_mask=trg_target_mask, src_padding_mask=src_padding_mask, trg_padding_mask=trg_padding_mask)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        src, trg = batch
        output = self.model(src, trg)
        output = output.view(-1, output.size(-1))
        loss = self.loss(output, trg)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output = self.model(src, trg)
        output = output.view(-1, output.size(-1))
        loss = self.loss(output, trg)
        return loss