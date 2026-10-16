"""
CODSOFT Task 2: Optional Module - Kaggle English Handwritten Characters Dataset
Module: kaggle_handwriting_analysis.py
Dataset: https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset
Description: Demonstrates the proper handling of 'english.csv' (3,410 character images across 62 classes)
             using a Convolutional Neural Network (CNN) for character recognition, and explains why
             treating it as a continuous text generation corpus is an architectural anti-pattern.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def analyze_kaggle_dataset(csv_path="english.csv"):
    """
    Inspects english.csv to verify distribution of 62 classes.
    """
    if not os.path.exists(csv_path):
        print(f"[Notice] Dataset file '{csv_path}' not found locally.")
        print("To run on Kaggle/Colab, download from: https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset")
        return None

    df = pd.read_csv(csv_path)
    print(f"Total dataset rows: {len(df):,}")
    print("Columns:", df.columns.tolist())
    print("\nSample records:")
    print(df.head(5))

    counts = df["label"].value_counts()
    print(f"\nUnique classes count: {len(counts)} (Expected: 62: 10 digits + 26 uppercase + 26 lowercase)")
    print(f"Samples per class: {counts.iloc[0]}")
    return df


def build_character_cnn(input_shape=(64, 64, 1), num_classes=62):
    """
    Proper model for 'english.csv': A Convolutional Neural Network (CNN)
    to classify or feature-extract individual handwritten character images.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name="Handwritten_Character_CNN")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("CODSOFT Task 2: Kaggle English Handwritten Characters Analysis")
    print("=" * 60)
    analyze_kaggle_dataset()
    model = build_character_cnn()
    model.summary()
