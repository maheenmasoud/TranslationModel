from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from prepare_data import create_dataloaders, load_data, train_bpe_tokenizer, encode_sentences, split_dataset
from model import Transformer
import pytorch_lightning as pl
from train import TransformerModel
from pytorch_lightning.loggers.wandb import WandbLogger

#no idea if this is what we should start with
def load_model(checkpoint_path, device='cpu'):
    model = TransformerModel.load_from_checkpoint(checkpoint_path)  # Replace with your model class
    model.eval()
    model.to(device)
    return model