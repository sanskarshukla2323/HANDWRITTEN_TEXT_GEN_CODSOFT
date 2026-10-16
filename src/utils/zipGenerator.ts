/**
 * Utility to generate and download the complete Python project package as a ZIP file.
 */
import JSZip from 'jszip';

export async function downloadPythonProjectZip(): Promise<void> {
  const zip = new JSZip();

  // Load or embed the Python project files
  const requirementsTxt = `# CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
tensorflow>=2.12.0
numpy>=1.23.5
pandas>=1.5.3
matplotlib>=3.7.1
streamlit>=1.28.0
Pillow>=9.5.0
`;

  const sampleText = `Handwriting is an intimate mirror of the human spirit. When ink touches paper, thought becomes tangible form. Each curve, stroke, and flourish carries the rhythm of breath and contemplation. Unlike mechanical typography, cursive manuscript reveals the subtle hesitation, speed, and emotion of the writer. Throughout the centuries, scholars kept leather journals filled with ink annotations. Every letter was crafted with intention, blending discipline and creative liberty. To write by hand is to converse slowly with oneself, leaving an enduring trace upon the world. The quick brown fox jumps over the lazy dog. Continuous practice allows the neural pathways of language and fine motor control to harmonize. As words unfold line after line, stories emerge from silent thoughts.`;

  const datasetPy = `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: dataset.py
"""
import os
import json
import numpy as np

def load_text(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    if filepath.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(filepath)
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise ValueError("No text column found in CSV file.")
        best_col = max(text_cols, key=lambda col: df[col].astype(str).str.len().mean())
        return " ".join(df[best_col].dropna().astype(str).tolist())
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def clean_text(raw_text: str, lowercase: bool = False) -> str:
    cleaned = raw_text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned

def build_vocabulary(text: str):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

def create_sequences(text: str, char_to_idx: dict, seq_length: int = 40, step: int = 3):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
    num_sequences = len(sentences)
    vocab_size = len(char_to_idx)
    X = np.zeros((num_sequences, seq_length, vocab_size), dtype=np.bool_)
    y = np.zeros((num_sequences, vocab_size), dtype=np.bool_)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1
    return X, y

def save_vocab(char_to_idx: dict, idx_to_char: dict, filepath: str):
    data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "vocab_size": len(char_to_idx)
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_vocab(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    char_to_idx = data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
    return char_to_idx, idx_to_char
`;

  const modelPy = `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: model.py
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_char_rnn(seq_length: int, vocab_size: int, hidden_units: int = 128, rnn_type: str = "LSTM", dropout: float = 0.2, learning_rate: float = 0.005):
    model = Sequential(name=f"CharLevel_{rnn_type}_Model")
    CellClass = LSTM if rnn_type.upper() == "LSTM" else SimpleRNN
    model.add(CellClass(hidden_units, input_shape=(seq_length, vocab_size)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model
`;

  const trainPy = `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: train.py
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import load_text, clean_text, build_vocabulary, create_sequences, save_vocab
from model import build_char_rnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sample_text.txt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--save_dir", type=str, default="saved_models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    text = clean_text(load_text(args.data))
    chars, char_to_idx, idx_to_char = build_vocabulary(text)
    save_vocab(char_to_idx, idx_to_char, os.path.join(args.save_dir, "vocab.json"))

    X, y = create_sequences(text, char_to_idx, seq_length=args.seq_length)
    model = build_char_rnn(args.seq_length, len(chars), hidden_units=args.hidden_units, rnn_type=args.rnn_type)

    history = model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs)
    model.save(os.path.join(args.save_dir, "final_char_rnn_model.keras"))

    plt.figure(figsize=(9, 4))
    plt.plot(range(1, args.epochs + 1), history.history['loss'], 'b-o', label='Training Loss')
    plt.title("Character-Level RNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "training_loss_curve.png"), dpi=200)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
`;

  const generatePy = `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: generate.py
"""
import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from dataset import load_vocab

def sample_with_temperature(predictions, temperature=0.7):
    preds = np.asarray(predictions).astype('float64')
    if temperature <= 0.01:
        return np.argmax(preds)
    preds = np.log(np.maximum(preds, 1e-12)) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, probs, 1)
    return int(np.argmax(probas))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="saved_models/final_char_rnn_model.keras")
    parser.add_argument("--vocab", type=str, default="saved_models/vocab.json")
    parser.add_argument("--seed", type=str, default="Handwriting is ")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    char_to_idx, idx_to_char = load_vocab(args.vocab)
    model = load_model(args.model)
    seq_length = model.input_shape[1]
    vocab_size = len(char_to_idx)

    sentence = args.seed[-seq_length:].rjust(seq_length)
    generated = args.seed

    for _ in range(args.length):
        x = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(sentence):
            if char in char_to_idx:
                x[0, t, char_to_idx[char]] = 1.0
        preds = model.predict(x, verbose=0)[0]
        next_idx = sample_with_temperature(preds, args.temperature)
        next_char = idx_to_char.get(next_idx, ' ')
        generated += next_char
        sentence = sentence[1:] + next_char

    print("\\n--- GENERATED TEXT ---")
    print(generated)

if __name__ == "__main__":
    main()
`;

  // Add files to root of the ZIP
  zip.file('requirements.txt', requirementsTxt);
  zip.file('sample_text.txt', sampleText);
  zip.file('dataset.py', datasetPy);
  zip.file('model.py', modelPy);
  zip.file('train.py', trainPy);
  zip.file('generate.py', generatePy);

  // Read README from server or embed
  zip.file('README.md', `# CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN

## Quick Start
1. Install requirements:
   pip install -r requirements.txt

2. Train the model:
   python train.py --data sample_text.txt --epochs 30 --hidden_units 128 --rnn_type LSTM

3. Generate text:
   python generate.py --seed "Handwriting is " --length 200 --temperature 0.7
`);

  // Generate zip blob
  const content = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(content);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'CODSOFT_Task2_Handwriting_Studio_Python_Project.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
