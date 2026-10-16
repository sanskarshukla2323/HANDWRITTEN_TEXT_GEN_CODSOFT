"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: generate.py
Description: Text generation pipeline utilizing a trained Character-Level RNN/LSTM
             with configurable seed text, generation length, and temperature scaling.
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from dataset import load_vocab
from model import load_trained_model


def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Applies temperature scaling to output logits / softmax probabilities.
    
    Formula:
        P(c_i) = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)
        
    - Low temperature (e.g. 0.2): Makes distribution peaky -> Conservative, repetitive, grammatically safe.
    - Medium temperature (e.g. 0.7): Balanced -> Natural variety and plausible vocabulary.
    - High temperature (e.g. 1.2+): Flattens distribution -> Unpredictable, creative, occasional typos.
    """
    preds = np.asarray(predictions).astype('float64')
    if temperature <= 0.01:
        return np.argmax(preds)

    preds = np.log(np.maximum(preds, 1e-12)) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, probs, 1)
    return int(np.argmax(probas))


def generate_text(
    model: tf.keras.Model,
    char_to_idx: dict,
    idx_to_char: dict,
    seed: str,
    gen_length: int = 200,
    temperature: float = 0.7
) -> str:
    """
    Generates text character-by-character starting from a seed phrase.
    """
    seq_length = model.input_shape[1]
    vocab_size = len(char_to_idx)

    # Pad or trim seed to match seq_length
    if len(seed) < seq_length:
        sentence = seed.rjust(seq_length)
    else:
        sentence = seed[-seq_length:]

    generated = seed
    print(f"\n[Generation] Priming with seed: '{seed}'")
    print(f"[Generation] Generating {gen_length} characters at temperature {temperature}...\n")

    for step in range(gen_length):
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(sentence):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char.get(next_idx, ' ')

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained Character-Level RNN.")
    parser.add_argument("--model", type=str, default="saved_models/final_char_rnn_model.keras", help="Path to trained model")
    parser.add_argument("--vocab", type=str, default="saved_models/vocab.json", help="Path to vocab JSON mapping")
    parser.add_argument("--seed", type=str, default="Handwriting is ", help="Initial starting phrase")
    parser.add_argument("--length", type=int, default=200, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.1 to 1.5)")
    parser.add_argument("--output", type=str, default=None, help="Optional text file to save generated output")
    args = parser.parse_args()

    # Verify model and vocab exist
    if not os.path.exists(args.model):
        print(f"\n[Error] Model file '{args.model}' not found!")
        print("Please train a model first using: python train.py --data sample_text.txt --epochs 30")
        return

    if not os.path.exists(args.vocab):
        print(f"\n[Error] Vocab file '{args.vocab}' not found!")
        return

    char_to_idx, idx_to_char = load_vocab(args.vocab)
    model = load_trained_model(args.model)

    generated_output = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seed=args.seed,
        gen_length=args.length,
        temperature=args.temperature
    )

    print("=" * 60)
    print("GENERATED RESULT:")
    print("=" * 60)
    print(generated_output)
    print("=" * 60)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(generated_output)
        print(f"[Export] Saved generated text to: {args.output}")


if __name__ == "__main__":
    main()
