"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Module: model.py
Description: TensorFlow / Keras neural network architectures for Character-Level RNN & LSTM.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop


def build_char_rnn(
    seq_length: int,
    vocab_size: int,
    hidden_units: int = 128,
    rnn_type: str = "LSTM",
    num_layers: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 0.005
) -> Sequential:
    """
    Constructs and compiles a Character-Level Recurrent Neural Network.
    
    Parameters:
    -----------
    seq_length : int
        The length of input character sequences (sliding window size).
    vocab_size : int
        The number of unique characters in the dataset vocabulary.
    hidden_units : int
        Number of hidden units in each recurrent layer (e.g. 64, 128, 256).
    rnn_type : str
        Either 'LSTM' (Long Short-Term Memory) or 'SimpleRNN'.
    num_layers : int
        Number of recurrent layers to stack.
    dropout : float
        Dropout rate to prevent overfitting on small text corpora.
    learning_rate : float
        Learning rate for the Adam optimizer.
        
    Returns:
    --------
    model : Sequential
        Compiled TensorFlow/Keras model ready for training.
    """
    model = Sequential(name=f"CharLevel_{rnn_type}_Model")

    # Select Recurrent Cell Class
    CellClass = LSTM if rnn_type.upper() == "LSTM" else SimpleRNN

    # First Recurrent Layer
    if num_layers == 1:
        model.add(CellClass(
            hidden_units,
            input_shape=(seq_length, vocab_size),
            return_sequences=False,
            name=f"{rnn_type.lower()}_layer_1"
        ))
        if dropout > 0:
            model.add(Dropout(dropout, name="dropout_1"))
    else:
        # Multi-layer stacked RNN
        for l_idx in range(num_layers):
            is_last = (l_idx == num_layers - 1)
            if l_idx == 0:
                model.add(CellClass(
                    hidden_units,
                    input_shape=(seq_length, vocab_size),
                    return_sequences=not is_last,
                    name=f"{rnn_type.lower()}_layer_{l_idx + 1}"
                ))
            else:
                model.add(CellClass(
                    hidden_units,
                    return_sequences=not is_last,
                    name=f"{rnn_type.lower()}_layer_{l_idx + 1}"
                ))
            if dropout > 0:
                model.add(Dropout(dropout, name=f"dropout_{l_idx + 1}"))

    # Dense Output Layer: Softmax over the character vocabulary
    model.add(Dense(vocab_size, activation="softmax", name="output_dense"))

    # Compile with Adam optimizer and Categorical Cross-Entropy loss
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    print(f"\n[Model Architecture] Created {rnn_type} model with {hidden_units} hidden units.")
    model.summary()
    return model


def save_trained_model(model: Sequential, filepath: str):
    """
    Saves the trained Keras model to disk (.keras or .h5 format).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"[Model] Saved trained weights to: {filepath}")


def load_trained_model(filepath: str) -> Sequential:
    """
    Loads a saved Keras model from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    model = keras_load_model(filepath)
    print(f"[Model] Successfully loaded trained model from: {filepath}")
    return model
