import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import random
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

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
                #print(f"Warning: Line {line_num} is malformed: {line}. Parts are: {parts}. Trying to split by space.")
                parts = line.split(' ')  # Try tab separator
                if len(parts) != 2:
                    #print(f"Error: Line {line_num} is still malformed: {line}. Parts are: {parts}")
                    continue
                else:
                    print(f"Successfully split line {line_num} by space.")
            eng, fre = parts
            english_sentences.append(eng)
            french_sentences.append(fre)

    return english_sentences, french_sentences

# Step 2: Train SentencePiece Tokenizers
def train_sentencepiece(sentences, model_prefix, vocab_size=1000, character_coverage=1.0):
    """
    Trains a SentencePiece model.

    Args:
        sentences (list): List of sentences to train the tokenizer.
        model_prefix (str): Prefix for the output model files.
        vocab_size (int): Vocabulary size.
        character_coverage (float): Coverage of characters (1.0 means full coverage).

    Returns:
        spm.SentencePieceProcessor: Trained SentencePiece tokenizer.
    """
    temp_file = f"{model_prefix}_raw.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='bpe',
        pad_id=2,       # <pad>
        unk_id=3,       # <unk>
        bos_id=0,       # <sos>
        eos_id=1,       # <eos>
        user_defined_symbols=["<mask>"],
    )
    
    os.remove(temp_file)
    print(f"Trained and saved {model_prefix}.model and {model_prefix}.vocab")
    
    # Load the trained model
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{model_prefix}.model")
    return tokenizer

# Step 3: Save Vocabulary
def save_vocab(tokenizer, vocab_file):
    """
    Saves the vocabulary with correct token-ID mappings.

    Args:
        tokenizer (spm.SentencePieceProcessor): Trained SentencePiece tokenizer.
        vocab_file (str): Path to save the vocabulary JSON file.
    """
    vocab = {
        "<sos>": tokenizer.bos_id(),
        "<eos>": tokenizer.eos_id(),
        "<pad>": tokenizer.pad_id(),
        "<unk>": tokenizer.unk_id(),
        "<mask>": tokenizer.piece_to_id("<mask>"),
    }

    for id in range(tokenizer.get_piece_size()):
        token = tokenizer.id_to_piece(id)
        if token not in vocab:  # Avoid overwriting special tokens
            vocab[token] = -id

    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Saved vocabulary to {vocab_file}.")


# Step 4: Encode Sentences
def encode_sentences(sentences, tokenizer, add_bos_eos=False):
    """
    Encodes sentences into lists of token IDs.

    Args:
        sentences (list): List of sentences to encode.
        tokenizer (spm.SentencePieceProcessor): Trained SentencePiece tokenizer.
        add_bos_eos (bool): Whether to add <sos> and <eos> tokens.

    Returns:
        list: List of encoded sentences as lists of token IDs.
    """
    encoded = []
    for sentence in sentences:
        ids = tokenizer.encode(sentence, out_type=int)
        if add_bos_eos:
            ids = [tokenizer.bos_id()] + ids + [tokenizer.eos_id()]
        encoded.append(ids)
    return encoded

# Step 5: Split Dataset
def split_dataset(encoded_eng, encoded_fre, test_ratio=0.1, val_ratio=0.2):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        encoded_eng (list): Encoded English sentences.
        encoded_fre (list): Encoded French sentences.
        test_ratio (float): Proportion of the dataset to include in the test split.
        val_ratio (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: Split datasets.
    """
    # First split into train_val and test
    train_val_eng, test_eng, train_val_fre, test_fre = train_test_split(
        encoded_eng,
        encoded_fre,
        test_size=test_ratio,
        random_state=42
    )
    
    # Then split train_val into train and val
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    train_eng, val_eng, train_fre, val_fre = train_test_split(
        train_val_eng,
        train_val_fre,
        test_size=val_ratio_adjusted,
        random_state=42
    )
    
    print(f"Training set size: {len(train_eng)}")
    print(f"Validation set size: {len(val_eng)}")
    print(f"Test set size: {len(test_eng)}")
    
    return train_eng, val_eng, test_eng, train_fre, val_fre, test_fre

# Step 6: Create Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences):
        # Ensure the source and target sentences are the same length
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
    """
    Generates a square mask for the target sequence.
    """
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))    # Convert ones to 0
    return mask

def collate_fn(batch):
    """
    Collate function to be used with DataLoader for padding.
    """
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=2, batch_first=True)  # <pad> token ID = 2
    trg_padded = pad_sequence(trg_batch, padding_value=2, batch_first=True)
    
    # Create padding masks
    src_padding_mask = (src_padded == 2).bool()  # <pad> token
    trg_padding_mask = (trg_padded == 2).bool()
    
    # Create target masks
    trg_mask = get_tgt_mask(trg_padded.size(1))
    
    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_padding_mask': src_padding_mask,
        'trg_padding_mask': trg_padding_mask,
        'trg_mask': trg_mask
    }

def create_dataloaders(train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=32):
    """
    Creates DataLoaders for training, validation, and testing datasets.

    Args:
        train_eng (list): Encoded training English sentences.
        val_eng (list): Encoded validation English sentences.
        test_eng (list): Encoded test English sentences.
        train_fre (list): Encoded training French sentences.
        val_fre (list): Encoded validation French sentences.
        test_fre (list): Encoded test French sentences.
        batch_size (int): Batch size.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
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
    """
    Saves tokenized sentences to a file, one sentence per line with token IDs separated by spaces.

    Args:
        tokenized_sentences (list): List of tokenized sentences (lists of token IDs).
        file_path (str): Path to the output file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in tokenized_sentences:
            sentence_str = ' '.join(map(str, sentence))
            f.write(sentence_str + '\n')
    print(f"Saved tokenized data to {file_path}.")

def main():
    # Parameters
    data_file = 'data.txt'  
    separator = '\t'
    vocab_size = 1000           # Vocabulary size
    batch_size = 32
    
    # Step 1: Load Data
    english_sentences, french_sentences = load_data(data_file, separator)
    print(f"Loaded {len(english_sentences)} sentence pairs.")
    
    # Step 2: Train SentencePiece Tokenizers
    eng_tokenizer = train_sentencepiece(english_sentences, "english", vocab_size=vocab_size)
    fre_tokenizer = train_sentencepiece(french_sentences, "french", vocab_size=vocab_size)
    
    # Step 3: Save Vocabularies
    save_vocab(eng_tokenizer, 'english_vocab.json')
    save_vocab(fre_tokenizer, 'french_vocab.json')
    
    # Step 4: Encode Sentences
    # Note: add_bos_eos is handled in collate_fn, so no need to add here
    encoded_english = encode_sentences(english_sentences, eng_tokenizer, add_bos_eos=False)
    encoded_french = encode_sentences(french_sentences, fre_tokenizer, add_bos_eos=False)
    print(f"Encoded English sentences: {len(encoded_english)}")
    print(f"Encoded French sentences: {len(encoded_french)}")
    
    # Step 5: Split Dataset
    train_eng, val_eng, test_eng, train_fre, val_fre, test_fre = split_dataset(encoded_english, encoded_french)
    
    # Step 6: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_eng, val_eng, test_eng, train_fre, val_fre, test_fre, batch_size=batch_size
    )
    
    # Step 7: Save Tokenized Data
    save_tokenized_data(train_eng, 'train_eng_tokenized.txt')
    save_tokenized_data(train_fre, 'train_fre_tokenized.txt')
    save_tokenized_data(val_eng, 'val_eng_tokenized.txt')
    save_tokenized_data(val_fre, 'val_fre_tokenized.txt')
    save_tokenized_data(test_eng, 'test_eng_tokenized.txt')
    save_tokenized_data(test_fre, 'test_fre_tokenized.txt')
    
    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    main()
