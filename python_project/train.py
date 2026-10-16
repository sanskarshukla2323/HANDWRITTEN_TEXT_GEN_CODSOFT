"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: train.py
Description: Full training pipeline with real-time loss tracking, epoch sample generation,
             matplotlib loss-curve plotting, and model saving.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from dataset import load_text, clean_text, build_vocabulary, create_sequences, save_vocab
from model import build_char_rnn, save_trained_model


class GenerateTextCallback(Callback):
    """
    A custom Keras callback that generates sample text at the end of every N epochs
    to visualize how the character-level model learns grammar and words over time.
    """
    def __init__(self, seed: str, char_to_idx: dict, idx_to_char: dict, seq_length: int, interval: int = 5):
        super().__init__()
        self.seed = seed
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.seq_length = seq_length
        self.interval = interval
        self.vocab_size = len(char_to_idx)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0 or epoch == 0:
            print(f"\n--- [Epoch {epoch + 1}] Sample Generation (Temp=0.5) ---")
            generated = self.seed
            sentence = self.seed[-self.seq_length:]
            if len(sentence) < self.seq_length:
                sentence = sentence.rjust(self.seq_length)

            for _ in range(80):
                x_pred = np.zeros((1, self.seq_length, self.vocab_size))
                for t, char in enumerate(sentence):
                    if char in self.char_to_idx:
                        x_pred[0, t, self.char_to_idx[char]] = 1.0

                preds = self.model.predict(x_pred, verbose=0)[0]
                # Temperature scaled sampling
                preds = np.asarray(preds).astype('float64')
                preds = np.log(np.maximum(preds, 1e-12)) / 0.5
                exp_preds = np.exp(preds)
                probs = exp_preds / np.sum(exp_preds)
                probas = np.random.multinomial(1, probs, 1)
                next_index = np.argmax(probas)
                next_char = self.idx_to_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            print(f"Seed: '{self.seed}'")
            print(f"Generated text: {generated}\n" + "-" * 50)


def plot_and_save_loss(history, save_path: str):
    """
    Generates and saves a clean training loss curve using Matplotlib.
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.plot(epochs, history.history['loss'], 'b-o', label='Training Loss (Categorical Cross-Entropy)', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
        
    plt.title('Character-Level RNN: Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Results] Loss graph saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Character-Level RNN on text.")
    parser.add_argument("--data", type=str, default="sample_text.txt", help="Path to input text or CSV file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for gradient descent")
    parser.add_argument("--seq_length", type=int, default=30, help="Input sequence length (window size)")
    parser.add_argument("--step", type=int, default=2, help="Stride step for sequence sliding window")
    parser.add_argument("--hidden_units", type=int, default=128, help="Number of recurrent hidden units")
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "SimpleRNN"], help="RNN cell type")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save model artifacts")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("CODSOFT Task 2: Character-Level RNN Training Pipeline")
    print("=" * 60)

    # 1. Load and clean raw text
    print(f"\n1. Loading dataset from: {args.data}")
    raw_text = load_text(args.data)
    text = clean_text(raw_text)
    print(f"Cleaned text preview (first 120 chars):\n'{text[:120]}...'")

    # 2. Build Character Vocabulary
    print("\n2. Building Character Vocabulary")
    chars, char_to_idx, idx_to_char = build_vocabulary(text)
    vocab_size = len(chars)

    vocab_file = os.path.join(args.save_dir, "vocab.json")
    save_vocab(char_to_idx, idx_to_char, vocab_file)

    # 3. Create Overlapping Sequences (X, y)
    print(f"\n3. Creating Sequence Sliding Windows (seq_len={args.seq_length}, step={args.step})")
    X, y = create_sequences(text, char_to_idx, seq_length=args.seq_length, step=args.step)
    print(f"Input tensor X shape: {X.shape}")
    print(f"Target tensor y shape: {y.shape}")

    # 4. Build Model
    print("\n4. Initializing Recurrent Model")
    model = build_char_rnn(
        seq_length=args.seq_length,
        vocab_size=vocab_size,
        hidden_units=args.hidden_units,
        rnn_type=args.rnn_type,
        dropout=0.2,
        learning_rate=args.lr
    )

    # 5. Training Callbacks
    seed_phrase = text[:args.seq_length] if len(text) >= args.seq_length else "Handwriting"
    text_gen_callback = GenerateTextCallback(
        seed=seed_phrase,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seq_length=args.seq_length,
        interval=max(1, args.epochs // 6)
    )

    model_checkpoint_path = os.path.join(args.save_dir, "best_char_rnn_model.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )

    # 6. Train Model
    print(f"\n5. Starting Training for {args.epochs} Epochs...")
    history = model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[text_gen_callback, checkpoint_callback]
    )

    # 7. Save Final Model
    final_model_path = os.path.join(args.save_dir, "final_char_rnn_model.keras")
    save_trained_model(model, final_model_path)

    # 8. Plot and Save Training Results
    loss_plot_path = os.path.join(args.save_dir, "training_loss_curve.png")
    plot_and_save_loss(history, loss_plot_path)

    history_csv_path = os.path.join(args.save_dir, "training_metrics.csv")
    history_df = pd.DataFrame({
        "epoch": range(1, args.epochs + 1),
        "loss": history.history["loss"],
        "accuracy": history.history.get("accuracy", [0] * args.epochs)
    })
    history_df.to_csv(history_csv_path, index=False)
    print(f"[Results] Training metrics exported to: {history_csv_path}")

    print("\n" + "=" * 60)
    print("Training Complete! Artifacts saved in:", args.save_dir)
    print(f"1. Model Weights: {final_model_path}")
    print(f"2. Vocab Mapping: {vocab_file}")
    print(f"3. Loss Curve:    {loss_plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
