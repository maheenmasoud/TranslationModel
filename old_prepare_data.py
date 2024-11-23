import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import random
from torch.nn.utils.rnn import pad_sequence

# Step 1: Load and Separate Data
def load_data(file_path, separator='\t'):
    english_sentences = []
    french_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(separator)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} is malformed: {line}")
                continue
            eng, fre = parts
            english_sentences.append(eng)
            french_sentences.append(fre)
    return english_sentences, french_sentences


# Step 2: Train BPE Tokenizers
# CITATION: https://pypi.org/project/tokenizers/
def train_bpe_tokenizer(sentences, model_name, vocab_size=1000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<sos>", "<eos>", "<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(f"{model_name}_tokenizer.json")
    print(f"Trained and saved {model_name} tokenizer.")
    return tokenizer


# Step 3: Save Vocabulary
def save_vocab(tokenizer, vocab_file):
    vocab = tokenizer.get_vocab()
    sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=4)
    print(f"Saved vocabulary to {vocab_file}.")


# Step 4: Encode Sentences
def encode_sentences(sentences, tokenizer):
    return [tokenizer.encode(sentence).ids for sentence in sentences]


# Step 5: Split Dataset
def split_dataset(encoded_eng, encoded_fre, test_ratio=0.1, val_ratio=0.2):
    # Ensure input lists are the same length
    assert len(encoded_eng) == len(encoded_fre), "Source and target sentences must be the same length."
    
    # Calculate dataset sizes
    total_size = len(encoded_eng)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size
    
    # Create indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split datasets based on indices
    train_eng = [encoded_eng[i] for i in train_indices]
    train_fre = [encoded_fre[i] for i in train_indices]
    val_eng = [encoded_eng[i] for i in val_indices]
    val_fre = [encoded_fre[i] for i in val_indices]
    test_eng = [encoded_eng[i] for i in test_indices]
    test_fre = [encoded_fre[i] for i in test_indices]
    
    print(f"Training set size: {len(train_eng)}")
    print(f"Validation set size: {len(val_eng)}")
    print(f"Test set size: {len(test_eng)}")
    
    return train_eng, val_eng, test_eng, train_fre, val_fre, test_fre


# Step 6: Create Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences):
        # Make sure the source and target sentences are the same length
        assert len(src_sentences) == len(trg_sentences), "Source and target sentences must be the same length."
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_sentences[idx], dtype=torch.long),
            'trg': torch.tensor(self.trg_sentences[idx], dtype=torch.long)
        }

def get_tgt_mask(size):
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    return mask
    
        
def collate_fn(batch):
    src_batch = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    trg_batch = [torch.tensor(torch.cat((torch.tensor([0]), item['trg'], torch.tensor([1])), dim=0), dtype=torch.long) for item in batch]  # Add <sos>=0, <eos>=1
    src_padded = pad_sequence(src_batch, padding_value=2, batch_first=True)  # Padding <pad>=2
    trg_padded = pad_sequence(trg_batch, padding_value=2, batch_first=True)
    
    src_padding_mask = (src_padded == 2).bool()  # Attention mask for source
    trg_padding_mask = (trg_padded == 2).bool()  # Attention mask for target

    trg_mask = get_tgt_mask(trg_padded.size(1))
    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_padding_mask': src_padding_mask,
        'trg_padding_mask': trg_padding_mask,
        "trg_mask" : trg_mask
    }


def create_dataloaders(train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=32):
    train_dataset = TranslationDataset(train_eng, train_fre)
    val_dataset = TranslationDataset(val_eng, val_fre)
    test_dataset = TranslationDataset(test_eng, test_fre)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        
    print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader


# Step 7: Save Tokenized Data
def save_tokenized_data(tokenized_sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in tokenized_sentences:
            sentence_str = ' '.join(map(str, sentence))
            f.write(sentence_str + '\n')
    print(f"Saved tokenized data to {file_path}.")


def main():
    # Parameters
    data_file = 'data.txt'  
    separator = '\t'         
    vocab_size = 1000           # Assumed size of the vocabulary
    batch_size = 32
    
    # Step 1: Load Data
    english_sentences, french_sentences = load_data(data_file, separator)
    print(f"Loaded {len(english_sentences)} sentence pairs.")
    
    # Step 2: Train BPE Tokenizers
    eng_tokenizer = train_bpe_tokenizer(english_sentences, "english", vocab_size=vocab_size)
    fre_tokenizer = train_bpe_tokenizer(french_sentences, "french", vocab_size=vocab_size)
    
    # Step 3: Save Vocabularies
    save_vocab(eng_tokenizer, 'english_vocab.json')
    save_vocab(fre_tokenizer, 'french_vocab.json')
    
    # Step 4: Encode Sentences
    encoded_english = encode_sentences(english_sentences, eng_tokenizer)
    encoded_french = encode_sentences(french_sentences, fre_tokenizer)
    print(f"Encoded English sentences: {len(encoded_english)}")
    print(f"Encoded French sentences: {len(encoded_french)}")
    
    # Step 5: Split Dataset
    train_eng, val_eng, test_eng, train_fre, val_fre, test_fre = split_dataset(encoded_english, encoded_french)
    
    # Step 6: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=batch_size
    )
    
    # Step 7: Save Tokenized Data (Optional)
    save_tokenized_data(train_eng, 'train_eng_tokenized.txt')
    save_tokenized_data(train_fre, 'train_fre_tokenized.txt')
    save_tokenized_data(val_eng, 'val_eng_tokenized.txt')
    save_tokenized_data(val_fre, 'val_fre_tokenized.txt')
    save_tokenized_data(test_eng, 'test_eng_tokenized.txt')
    save_tokenized_data(test_fre, 'test_fre_tokenized.txt')
    
    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    main()