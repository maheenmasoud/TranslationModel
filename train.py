from model import Transformer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class TransformerModel(pl.LightningModule):
    def __init__(self, 
                 model: Transformer,
                learning_rate: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, trg):
        return self.model(src, trg)
    
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

dataset = []
train_loader = DataLoader(dataset, batch_size=32)

model_1 = Transformer(d_model=512, n_heads=8, d_ff=2048)
model = TransformerModel(model_1, learning_rate=0.0001)

trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=train_loader)
