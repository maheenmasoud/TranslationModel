from model import Transformer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pytorch_lightning.loggers.wandb import WandbLogger
from nltk.translate.bleu_score import sentence_bleu
from prepare_data import create_dataloaders, load_data, train_bpe_tokenizer, encode_sentences, split_dataset
from torch.optim.lr_scheduler import StepLR

class TransformerModel(pl.LightningModule):
    def __init__(self, 
                 model: Transformer,
                learning_rate: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss(ignore_index=2)

    def forward(self, src, trg, src_padding_mask, trg_padding_mask, trg_mask):
        #run the transformer model with the data
        return self.model(src, trg, src_padding_mask=src_padding_mask, trg_padding_mask=trg_padding_mask, trg_mask=trg_mask)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
            },}
    
    def training_step(self, batch):
        src = batch['src']
        trg = batch['trg'] 
        src_padding_mask = batch['src_padding_mask']
        trg_padding_mask = batch['trg_padding_mask']
        trg_mask = batch['trg_mask']
        
        output = self.model(src, trg, src_padding_mask, trg_padding_mask, trg_mask)
        output = output.permute(1, 0, 2)

        output = output.reshape(-1, output.size(-1))  # Flatten output to [batch_size * seq_len, num_classes]
        trg = trg.reshape(-1)  # Flatten target to [batch_size * seq_len]

        loss = self.loss(output, trg)
        self.log('train_loss', loss, batch_size=32, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        src = batch['src']
        trg = batch['trg'] 
        src_padding_mask = batch['src_padding_mask']
        trg_padding_mask = batch['trg_padding_mask']
        trg_mask = batch['trg_mask']

        output = self.model(src, trg, src_padding_mask, trg_padding_mask, trg_mask)
        output = output.permute(1, 0, 2)

        #flatten output and trg
        loss = self.loss(output.reshape(-1, output.size(-1)), trg.reshape(-1))
        self.log('val_loss', loss, batch_size=32, on_epoch=True, prog_bar=True)

        #convert predicted and trg to list of vectors, with each vector being a sentence
        predicted = output.argmax(dim=-1)

        predicted = predicted.cpu().numpy().tolist()
        trg = trg.cpu().numpy().tolist()

        #calculate the bleu score and then log it
        bleu_score = self.calculate_bleu(predicted, trg)
        self.log('bleu_score', bleu_score) 
        return loss
    
    def calculate_bleu(self, predicted, target):
        bleu_scores = []
        for p, t in zip(predicted, target):
            # convert indices back to words, assuming trg and predicted are token indices
            p = [str(i) for i in p if i != 2]  # filter out padding tokens (index 2)
            t = [str(i) for i in t if i != 2]  # filter out padding tokens (index 2)
            bleu_scores.append(sentence_bleu([t], p))
        return sum(bleu_scores) / len(bleu_scores) 