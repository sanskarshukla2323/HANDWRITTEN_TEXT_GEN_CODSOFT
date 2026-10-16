"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: dataset.py
Description: Data loading, cleaning, character vocabulary creation, 
             sequence vectorization (X, y pairs), and serialization.
"""

import os
import json
import numpy as np


def load_text(filepath: str) -> str:
    """
    Reads and returns the text from a local text or CSV file.
    Gracefully handles encoding exceptions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Handle CSV or plain text
    if filepath.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(filepath)
        # Look for a text-like column or concatenate all text entries
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise ValueError("No text column found in CSV file.")
        # Choose the column with the highest average length
        best_col = max(text_cols, key=lambda col: df[col].astype(str).str.len().mean())
        print(f"[Dataset] Extracted text from CSV column: '{best_col}'")
        text = " ".join(df[best_col].dropna().astype(str).tolist())
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

    if len(text.strip()) == 0:
        raise ValueError(f"File '{filepath}' is empty or contains no readable text.")

    return text


def clean_text(raw_text: str, lowercase: bool = False, remove_non_ascii: bool = False) -> str:
    """
    Cleans text by normalizing whitespace, optional lowercasing, and filtering invalid characters.
    """
    cleaned = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    
    if lowercase:
        cleaned = cleaned.lower()
        
    if remove_non_ascii:
        cleaned = "".join([c for c in cleaned if ord(c) < 128])
        
    # Collapse excessive blank lines
    import re
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


def build_vocabulary(text: str):
    """
    Extracts unique characters and creates bidirectional mapping dictionaries:
    char_to_idx: Maps character symbol to integer index
    idx_to_char: Maps integer index to character symbol
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
    
    print(f"[Vocabulary] Total characters: {len(text):,}")
    print(f"[Vocabulary] Unique vocabulary size: {vocab_size} characters")
    return chars, char_to_idx, idx_to_char


def create_sequences(text: str, char_to_idx: dict, seq_length: int = 40, step: int = 3):
    """
    Splits text into overlapping chunks of length `seq_length` (X)
    and the subsequent next character as target (y).
    
    Example:
    Text: "Handwriting"
    If seq_length = 4, step = 1:
      Input (X): "Hand" -> Target (y): 'w'
      Input (X): "andw" -> Target (y): 'r'
    """
    sentences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
        
    num_sequences = len(sentences)
    print(f"[Sequences] Extracted {num_sequences:,} training sequence pairs (seq_len={seq_length}, step={step})")
    
    if num_sequences == 0:
        raise ValueError(
            f"Text length ({len(text)}) is too short for sequence length ({seq_length}). "
            "Please provide a longer text corpus."
        )

    # Vectorize into numpy arrays:
    # X: [num_sequences, seq_length, vocab_size] (One-hot encoded)
    # y: [num_sequences, vocab_size] (One-hot encoded target)
    vocab_size = len(char_to_idx)
    X = np.zeros((num_sequences, seq_length, vocab_size), dtype=np.bool_)
    y = np.zeros((num_sequences, vocab_size), dtype=np.bool_)
    
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1
        
    return X, y


def save_vocab(char_to_idx: dict, idx_to_char: dict, filepath: str):
    """
    Saves character mappings to a JSON file.
    """
    data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "vocab_size": len(char_to_idx)
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Vocabulary] Saved mapping to: {filepath}")


def load_vocab(filepath: str):
    """
    Loads character mappings from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Vocabulary mapping file not found at: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    char_to_idx = data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
    return char_to_idx, idx_to_char
