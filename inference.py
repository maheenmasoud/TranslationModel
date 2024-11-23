# inference.py

import os
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
import nltk
from train import TransformerModel  # Import the LightningModule
from model import Transformer
from tqdm import tqdm

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===============================
# Step 1: Define the Dataset
# ===============================

class TranslationTestDataset(Dataset):
    def __init__(self, src_file, trg_file):
        """
        Args:
            src_file (str): Path to the tokenized source sentences file.
            trg_file (str): Path to the tokenized target sentences file.
        """
        assert os.path.exists(src_file), f"Source file {src_file} not found."
        assert os.path.exists(trg_file), f"Target file {trg_file} not found."
        
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = [list(map(int, line.strip().split())) for line in f if line.strip()]
        
        with open(trg_file, 'r', encoding='utf-8') as f:
            self.trg_sentences = [list(map(int, line.strip().split())) for line in f if line.strip()]
        
        assert len(self.src_sentences) == len(self.trg_sentences), "Source and target files must have the same number of sentences."

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_sentences[idx], dtype=torch.long),
            'trg': torch.tensor(self.trg_sentences[idx], dtype=torch.long)
        }


# ===============================
# Step 2: Define Masking Functions
# ===============================

def get_tgt_mask(size):
    """
    Generates a square mask for the target sequence.
    """
    mask = torch.tril(torch.ones((size, size), device=device) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def collate_fn(batch, pad_id_src, pad_id_trg):
    """
    Collate function to be used with DataLoader for padding and masking.
    """
    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]
    
    # Assuming <sos> and <eos> are already handled during encoding
    # If not, you can add them here based on your tokenizer's configuration
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=pad_id_src, batch_first=True).to(device)
    trg_padded = pad_sequence(trg_batch, padding_value=pad_id_trg, batch_first=True).to(device)
    
    # Create padding masks
    src_padding_mask = (src_padded == pad_id_src).to(device)  # <pad> token
    trg_padding_mask = (trg_padded == pad_id_trg).to(device)
    
    # Create target masks
    trg_mask = get_tgt_mask(trg_padded.size(1))
    
    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_padding_mask': src_padding_mask,
        'trg_padding_mask': trg_padding_mask,
        'trg_mask': trg_mask
    }


# ===============================
# Step 3: Load Tokenizer
# ===============================

def load_tokenizer(model_path):
    """
    Loads a trained Hugging Face Tokenizer.

    Args:
        model_path (str): Path to the tokenizer JSON file.

    Returns:
        Tokenizer: Loaded tokenizer.
    """
    tokenizer = Tokenizer.from_file(model_path)
    return tokenizer


# ===============================
# Step 4: Decode Sentences
# ===============================

def decode_sentence(tokenizer, token_ids):
    """
    Decodes a list of token IDs to a sentence.

    Args:
        tokenizer (Tokenizer): Trained Hugging Face Tokenizer.
        token_ids (list): List of token IDs.

    Returns:
        str: Decoded sentence.
    """
    # Convert token IDs back to a string, skipping special tokens
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ===============================
# Step 5: Calculate BLEU Score
# ===============================

def calculate_bleu(references, hypotheses):
    """
    Calculates the BLEU score for the corpus.

    Args:
        references (list of str): Reference translations.
        hypotheses (list of str): Hypothesized translations.

    Returns:
        float: BLEU score.
    """
    # Tokenize references and hypotheses
    references = [[nltk.word_tokenize(ref)] for ref in references]
    hypotheses = [nltk.word_tokenize(hyp) for hyp in hypotheses]
    
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score


# ===============================
# Step 6: Run Inference
# ===============================

def run_inference(model, src_loader, fre_tokenizer, pad_id_src, pad_id_trg, target_vocab_size):
    """
    Runs inference on the test dataset and collects references and hypotheses.

    Args:
        model (TransformerModel): Trained translation model.
        src_loader (DataLoader): DataLoader for the test source sentences.
        fre_tokenizer (Tokenizer): French tokenizer.
        pad_id_src (int): <pad> token ID for source language.
        pad_id_trg (int): <pad> token ID for target language.
        target_vocab_size (int): Vocabulary size for the target language.

    Returns:
        list, list: Lists of reference sentences and hypothesis sentences.
    """
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(src_loader, desc="Running Inference")):
            print(f"Processing batch {i+1}/{len(src_loader)}")
            src = batch['src'].to(device)  # [batch_size, src_len]
            src_padding_mask = batch['src_padding_mask'].to(device)  # [batch_size, src_len]
            trg = batch['trg'].to(device)  # [batch_size, trg_len]
            trg_padding_mask = batch['trg_padding_mask'].to(device)  # [batch_size, trg_len]
            trg_mask = batch['trg_mask'].to(device)  # [trg_len, trg_len]
            
            # Generate translations using the model's generate method
            generated_ids = model.model.generate(
                src,
                src_padding_mask,
                fre_tokenizer,
                max_len=50,
                #pad_token_id=pad_id_trg,
                #eos_token_id=1  # Assuming <eos> token ID is 1
            )  # List of lists
            print(f"Generated IDs: {generated_ids}")
            
            # Reference sentences (ground truth)
            for trg_ids in trg:
                ref_sentence = decode_sentence(fre_tokenizer, trg_ids.tolist())
                references.append(ref_sentence)
            
            # Hypothesis sentences (model predictions)
            for hyp_ids in generated_ids:
                # Check if generated token IDs are within the target vocabulary
                if any(token_id >= target_vocab_size for token_id in hyp_ids):
                    print(f"Warning: Generated token ID {max(hyp_ids)} exceeds target_vocab_size {target_vocab_size}.")
                hyp_sentence = decode_sentence(fre_tokenizer, hyp_ids)
                hypotheses.append(hyp_sentence)
    
    return references, hypotheses


# ===============================
# Step 7: Main Function
# ===============================

def main():
    # Paths to necessary files
    test_src_file = 'test_eng_tokenized.txt'
    test_trg_file = 'test_fre_tokenized.txt'
    english_tokenizer_json = 'english_tokenizer.json'  # Updated to use tokenizer.json
    french_tokenizer_json = 'french_tokenizer.json'    # Updated to use tokenizer.json
    checkpoint_path = 'English_to_French_Translation/s4dr1017/checkpoints/epoch=1-step=8358.ckpt'  # Path to the model weights file
    
    # Model hyperparameters (should match the ones used during training)
    d_model = 512
    nhead = 8
    dim_feedforward = 512
    dropout = 0.1
    
    # Vocabulary sizes (should match the ones used during training)
    source_vocab_size = 1000
    target_vocab_size = 1000

    # Load Tokenizers from JSON files
    eng_tokenizer = load_tokenizer(english_tokenizer_json)
    fre_tokenizer = load_tokenizer(french_tokenizer_json)
    print("Loaded Hugging Face Tokenizers.")
    
    # Retrieve special token IDs
    pad_id_src = fre_tokenizer.token_to_id("<pad>")
    pad_id_trg = fre_tokenizer.token_to_id("<pad>")  # Assuming <pad> is same across languages
    print(f"<pad> token ID (source): {pad_id_src}")
    print(f"<pad> token ID (target): {pad_id_trg}")
    
    # Load Test Dataset
    test_dataset = TranslationTestDataset(test_src_file, test_trg_file)
    print(f"Loaded {len(test_dataset)} test sentence pairs.")
    
    # Create DataLoader for Test Set
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_id_src, pad_id_trg)
    )
    print("Created DataLoader for the test set.")
    
    # Initialize the Transformer Model
    model_1 = Transformer(
        d_model=d_model,
        nhead=nhead,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # Load Trained Model from '.ckpt' file using load_from_checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model weights file {checkpoint_path} not found.")
    
    # Load the TransformerModel from checkpoint
    try:
        model = TransformerModel.load_from_checkpoint(
            checkpoint_path,
            model=model_1,
            learning_rate=0.0001,
            d_model=d_model,
            nhead=nhead,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        print("Successfully loaded model weights from checkpoint.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Verify model parameters are loaded (add debugging)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    # Move the model to the appropriate device and set to evaluation mode
    model.to(device)
    model.eval()
    print(f"Model loaded and moved to {device}.")
    
    # Step 6: Run Inference
    references, hypotheses = run_inference(
        model,
        test_loader,
        fre_tokenizer,
        pad_id_src,
        pad_id_trg,
        target_vocab_size
    )
    print("Completed inference on the test set.")
    
    # Step 7: Calculate BLEU Score
    bleu = calculate_bleu(references, hypotheses)
    print(f"Corpus BLEU Score: {bleu * 100:.2f}")
    
    # Optional: Save the references and hypotheses for further analysis
    with open('test_references.txt', 'w', encoding='utf-8') as f:
        for ref in references:
            f.write(ref + '\n')
    
    with open('test_hypotheses.txt', 'w', encoding='utf-8') as f:
        for hyp in hypotheses:
            f.write(hyp + '\n')
    
    print("Saved references and hypotheses to 'test_references.txt' and 'test_hypotheses.txt'.")
    print("Inference and evaluation completed successfully.")


# ===============================
# Step 8: Execute the Script
# ===============================

if __name__ == '__main__':
    main()
