from prepare_data import create_dataloaders, load_data, train_bpe_tokenizer, encode_sentences, split_dataset
from model import Transformer
import pytorch_lightning as pl
from train import TransformerModel
from pytorch_lightning.loggers.wandb import WandbLogger
from nltk.translate.bleu_score import sentence_bleu

def make_data_loaders():

    data_file = 'data.txt'
    separator = '\t'
    vocab_size = 8000
    batch_size = 32

    english_sentences, french_sentences = load_data(data_file, separator)

    eng_tokenizer = train_bpe_tokenizer(english_sentences, "english", vocab_size=vocab_size)
    fre_tokenizer = train_bpe_tokenizer(french_sentences, "french", vocab_size=vocab_size)
    
    encoded_english = encode_sentences(english_sentences, eng_tokenizer)
    encoded_french = encode_sentences(french_sentences, fre_tokenizer)

    train_eng, val_eng, test_eng, train_fre, val_fre, test_fre = split_dataset(encoded_english, encoded_french)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=batch_size
    )

    print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = make_data_loaders()

    source_vocab_size = 1000
    target_vocab_size = 1000
    
    model_1 = Transformer(
        d_model=512,
        nhead=8,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
    )
    model = TransformerModel(model_1, learning_rate=0.0001)

    logger = WandbLogger(project='English_to_French_Translation', log_model=True)

    trainer = pl.Trainer(max_epochs=10, logger=logger, accelerator="auto", devices=1, callbacks=pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1))
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    main()