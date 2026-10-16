import React, { useState } from 'react';
import { Code2, Copy, Check, Download, ExternalLink, Terminal, FileCode, CheckCircle2 } from 'lucide-react';
import { downloadPythonProjectZip } from '../../utils/zipGenerator';

export const PythonExportTab: React.FC = () => {
  const [activeFile, setActiveFile] = useState<string>('train.py');
  const [copied, setCopied] = useState<boolean>(false);
  const [isZipping, setIsZipping] = useState<boolean>(false);

  const fileContents: Record<string, { desc: string; code: string }> = {
    'train.py': {
      desc: 'Complete training pipeline: builds sequence windows, trains model, plots Matplotlib loss curve, and saves checkpoints.',
      code: `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: train.py
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import load_text, clean_text, build_vocabulary, create_sequences, save_vocab
from model import build_char_rnn, save_trained_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sample_text.txt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "SimpleRNN"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    text = clean_text(load_text(args.data))
    chars, char_to_idx, idx_to_char = build_vocabulary(text)
    save_vocab(char_to_idx, idx_to_char, os.path.join(args.save_dir, "vocab.json"))

    X, y = create_sequences(text, char_to_idx, seq_length=args.seq_length)
    model = build_char_rnn(args.seq_length, len(chars), hidden_units=args.hidden_units, rnn_type=args.rnn_type, learning_rate=args.lr)

    history = model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs)
    save_trained_model(model, os.path.join(args.save_dir, "final_char_rnn_model.keras"))

    # Plot training loss graph
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, args.epochs + 1), history.history['loss'], 'b-o', label='Categorical Cross-Entropy Loss')
    plt.title("Character-Level RNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_loss_curve.png"), dpi=200)
    print("Training finished successfully! Artifacts saved to:", args.save_dir)

if __name__ == "__main__":
    main()`
    },
    'model.py': {
      desc: 'Keras neural network architecture defining SimpleRNN and LSTM recurrent layers and softmax dense projection.',
      code: `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: model.py
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_char_rnn(seq_length: int, vocab_size: int, hidden_units: int = 128, rnn_type: str = "LSTM", dropout: float = 0.2, learning_rate: float = 0.005) -> Sequential:
    model = Sequential(name=f"CharLevel_{rnn_type}_Model")
    CellClass = LSTM if rnn_type.upper() == "LSTM" else SimpleRNN
    model.add(CellClass(hidden_units, input_shape=(seq_length, vocab_size)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model

def save_trained_model(model: Sequential, filepath: str):
    model.save(filepath)

def load_trained_model(filepath: str) -> Sequential:
    return keras_load(filepath)`
    },
    'dataset.py': {
      desc: 'Handles text and CSV reading, character vocabulary indexing, and sliding-window one-hot sequence vectorization.',
      code: `"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: dataset.py
"""
import os
import json
import numpy as np

def load_text(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if filepath.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(filepath)
        text_cols = [c for c in df.columns if df[c].dtype == object]
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

def create_sequences(text: str, char_to_idx: dict, seq_length: int = 30, step: int = 2):
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
    data = {"char_to_idx": char_to_idx, "idx_to_char": {str(k): v for k, v in idx_to_char.items()}}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_vocab(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["char_to_idx"], {int(k): v for k, v in data["idx_to_char"].items()}`
    },
    'generate.py': {
      desc: 'Text generation engine with temperature-scaled multinomial sampling and seed phrase priming.',
      code: `"""
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

    print("\\n--- GENERATED RESULT ---")
    print(generated)`
    },
    'streamlit_app.py': {
      desc: 'Interactive multi-page Streamlit web application matching all 10 CODSOFT features.',
      code: `# Complete Streamlit application with Dataset upload, live training, loss curve, and generation.
# Run with: streamlit run streamlit_app.py
# (View full 300+ line implementation inside the python_project/ folder or download project zip)`
    },
    'requirements.txt': {
      desc: 'Pinned Python library dependencies for standard environments and Google Colab.',
      code: `tensorflow>=2.12.0
numpy>=1.23.5
pandas>=1.5.3
matplotlib>=3.7.1
streamlit>=1.28.0
Pillow>=9.5.0`
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(fileContents[activeFile]?.code || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadZip = async () => {
    try {
      setIsZipping(true);
      await downloadPythonProjectZip();
    } finally {
      setIsZipping(false);
    }
  };

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white">07. Python Source Code & Google Colab Pipeline</h2>
          <p className="text-xs text-slate-400 mt-1">
            Complete standalone Python scripts for local execution or running in Google Colab with free T4 GPU acceleration.
          </p>
        </div>

        <button
          onClick={handleDownloadZip}
          disabled={isZipping}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer whitespace-nowrap shrink-0"
        >
          <Download className="w-3.5 h-3.5" />
          <span>{isZipping ? 'Bundling Zip...' : 'Download Project Package (.zip)'}</span>
        </button>
      </div>

      {/* Google Colab 3-Step Execution Box */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Terminal className="w-4 h-4 text-indigo-400" />
            <span>How to Run in Google Colab (Zero-Setup Cloud GPU)</span>
          </h3>
          <a
            href="https://colab.research.google.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-semibold"
          >
            <span>Open Google Colab</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
            <span className="text-indigo-400 font-bold block">Step 1: Upload Files</span>
            <span className="text-slate-300">
              Download the project zip, extract it, and drag the files into your Colab session files panel.
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
            <span className="text-indigo-400 font-bold block">Step 2: Train Model</span>
            <div className="font-mono-code text-[11px] text-slate-300 bg-slate-950 p-1.5 rounded">
              !python train.py --epochs 30 --rnn_type LSTM
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
            <span className="text-indigo-400 font-bold block">Step 3: Generate Text</span>
            <div className="font-mono-code text-[11px] text-slate-300 bg-slate-950 p-1.5 rounded">
              !python generate.py --temperature 0.7
            </div>
          </div>
        </div>
      </div>

      {/* Code Browser */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        {/* File Tabs */}
        <div className="flex flex-wrap items-center gap-1.5 border-b border-slate-700/60 pb-3">
          {Object.keys(fileContents).map((fileName) => {
            const isActive = activeFile === fileName;
            return (
              <button
                key={fileName}
                onClick={() => setActiveFile(fileName)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono-code transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-slate-800 text-indigo-400 font-bold border border-slate-700'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <FileCode className="w-3.5 h-3.5" />
                <span>{fileName}</span>
              </button>
            );
          })}
        </div>

        {/* File Description & Copy */}
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>{fileContents[activeFile]?.desc}</span>
          <button
            onClick={handleCopyCode}
            className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-xs transition-colors cursor-pointer"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy Code'}</span>
          </button>
        </div>

        {/* Code Content */}
        <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono-code text-xs text-slate-200 overflow-x-auto leading-relaxed max-h-[450px]">
          {fileContents[activeFile]?.code}
        </pre>
      </div>

    </div>
  );
};
