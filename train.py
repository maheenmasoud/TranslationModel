from model import Transformer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pytorch_lightning.loggers.wandb import WandbLogger
from nltk.translate.bleu_score import sentence_bleu
from prepare_data import create_dataloaders, load_data, train_bpe_tokenizer, encode_sentences, split_dataset

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
        self.validation_step((src,trg))
        return self.model(src, trg, trg_mask=trg_target_mask, src_padding_mask=src_padding_mask, trg_padding_mask=trg_padding_mask)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch):
        src = batch['src']
        trg = batch['trg'] 

        output = self.model(src, trg)
        output = output.view(-1, output.size(-1))
        loss = self.loss(output, trg)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        src = batch['src']
        trg = batch['trg'] 

        output = self.model(src, trg)
        output = output.view(-1, output.size(-1))
        loss = self.loss(output, trg)
        self.log('val_loss', loss)
        #convert predicted and trg to list of vectors, with each vector being a sentence
        predicted = output.argmax(dim=-1)
        predicted = predicted.cpu().numpy().tolist()
        trg = trg.cpu().numpy().tolist()

        #calculate the bleu score and then log it
        bleu_score = self.calculate_bleu(predicted, trg)
        self.log('val_bleu', bleu_score) 
        return loss
    
    def calculate_bleu(self, predicted, target):
        bleu_scores = []
        for p, t in zip(predicted, target):
            # Convert indices back to words, assuming trg and predicted are token indices
            p = [str(i) for i in p if i != 0]  # Filter out padding tokens (index 0)
            t = [str(i) for i in t if i != 0]  # Filter out padding tokens (index 0)
            bleu_scores.append(sentence_bleu([t], p))
        return sum(bleu_scores) / len(bleu_scores)  # Average BLEU score

# def make_data_loaders():

#     data_file = 'data.txt'
#     separator = '\t'
#     vocab_size = 8000
#     batch_size = 32

#     english_sentences, french_sentences = load_data(data_file, separator)

#     eng_tokenizer = train_bpe_tokenizer(english_sentences, "english", vocab_size=vocab_size)
#     fre_tokenizer = train_bpe_tokenizer(french_sentences, "french", vocab_size=vocab_size)
    
#     encoded_english = encode_sentences(english_sentences, eng_tokenizer)
#     encoded_french = encode_sentences(french_sentences, fre_tokenizer)

#     train_eng, val_eng, test_eng, train_fre, val_fre, test_fre = split_dataset(encoded_english, encoded_french)

#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=batch_size
#     )

#     print("DataLoaders created successfully.")
#     return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = make_data_loaders()
# source_vocab_size = 1000
# target_vocab_size = 1000
# model = Transformer(
#         d_model=512,
#         nhead=8,
#         source_vocab_size=source_vocab_size,
#         target_vocab_size=target_vocab_size,
#     )

# for batch_idx, batch in enumerate(train_loader):
#     # Process the batch
#     test = batch
#     # Break after processing the first batch
#     break
# src, trg = batch
# print(src)
# # model(train_loader)