"""
CODSOFT Task 2: Handwritten Text Generation using Character-Level RNN
Streamlit Web Application: Handwriting Studio
"""

import os
import io
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="Handwriting Studio | CODSOFT Task 2",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "model" not in st.session_state:
    st.session_state.model = None
if "char_to_idx" not in st.session_state:
    st.session_state.char_to_idx = None
if "idx_to_char" not in st.session_state:
    st.session_state.idx_to_char = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

# Sidebar Navigation
st.sidebar.title("✍️ Handwriting Studio")
st.sidebar.caption("CODSOFT AI/ML Internship · Task 2")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Project Overview",
        "2. Dataset & Preprocessing",
        "3. Model Architecture & Training",
        "4. Text Generation",
        "5. Kaggle Dataset Analysis",
        "6. Model Management & Downloads"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Internship Info:**")
st.sidebar.text("Track: AI/ML Engineering\nTask: #2 Character-Level RNN")

# -------------------------------------------------------------
# 1. Project Overview
# -------------------------------------------------------------
if app_mode == "1. Project Overview":
    st.title("🖋️ Handwriting Studio: Character-Level RNN")
    st.markdown("### Submission-Ready Project for CODSOFT AI/ML Internship (Task 2)")
    
    st.info(
        "**Main Objective:** Build a working application that demonstrates character-level text generation "
        "using a genuine Recurrent Neural Network (RNN/LSTM). The network learns sequence probabilities "
        "from training text and generates new sentences character by character."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Architecture", value="LSTM / SimpleRNN")
    with col2:
        st.metric(label="Granularity", value="Character-Level")
    with col3:
        st.metric(label="Sampling Logic", value="Temperature Softmax")

    st.markdown("""
    #### 🧠 How Character-Level RNNs Work:
    1. **Tokenization at Character Level:** Instead of splitting into words, the vocabulary consists of discrete symbols:
       letters (`a-z`, `A-Z`), numbers (`0-9`), punctuation (`.`, `,`, `!`), and spaces.
    2. **Sequence Windows ($X \\to y$):** Given a window of length $L$ characters:
       $$[c_1, c_2, \\dots, c_L] \\longrightarrow c_{L+1}$$
    3. **Hidden State Recursion:** At each time step $t$, the recurrent cell updates its memory state:
       $$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
    4. **Softmax Output Logits:** The final hidden state projects across the vocabulary size $V$:
       $$P(c_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$
       where $T$ represents the **Temperature** parameter.
    
    #### 📝 Instructions for Reviewers:
    1. Navigate to **2. Dataset & Preprocessing** to upload text or load the default corpus.
    2. Build the character vocabulary and prepare the training sliding windows.
    3. Head to **3. Model Architecture & Training** to configure and train the network.
    4. Use **4. Text Generation** to generate novel text with temperature sliders.
    5. Review **5. Kaggle Dataset Analysis** to understand the distinction between text sequence models and the Kaggle handwritten characters dataset.
    """)

# -------------------------------------------------------------
# 2. Dataset & Preprocessing
# -------------------------------------------------------------
elif app_mode == "2. Dataset & Preprocessing":
    st.title("📂 Dataset Ingestion & Preprocessing")
    st.write("Upload a custom training text file, CSV dataset, or load a built-in literature corpus.")

    data_source = st.radio("Choose Dataset Source", ["Use Built-in Sample Corpus", "Upload Custom Text File (.txt)", "Upload CSV File (.csv)"])

    if data_source == "Use Built-in Sample Corpus":
        sample_choice = st.selectbox(
            "Select Sample Corpus",
            [
                "The Art of Handwriting & The Human Mind (Essay)",
                "Alice in Wonderland Excerpt (Literature)",
                "Recurrent Neural Networks (Machine Learning Notes)"
            ]
        )
        if "Handwriting" in sample_choice:
            sample_text = (
                "Handwriting is an intimate mirror of the human spirit. When ink touches paper, thought becomes tangible form. "
                "Each curve, stroke, and flourish carries the rhythm of breath and contemplation. Unlike mechanical typography, "
                "cursive manuscript reveals the subtle hesitation, speed, and emotion of the writer. Throughout the centuries, "
                "scholars kept leather journals filled with ink annotations. Every letter was crafted with intention, blending "
                "discipline and creative liberty. To write by hand is to converse slowly with oneself, leaving an enduring trace."
            )
        elif "Alice" in sample_choice:
            sample_text = (
                "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or "
                "twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and "
                "what is the use of a book, thought Alice without pictures or conversations? So she was considering in her own "
                "mind as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a "
                "daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran by."
            )
        else:
            sample_text = (
                "Recurrent neural networks process sequential data by maintaining an internal hidden state. At each time step, "
                "the network consumes the current character input vector and updates its hidden representation. In character-level "
                "language modeling, the vocabulary consists of discrete characters. The network minimizes categorical cross-entropy loss."
            )
        st.session_state.raw_text = sample_text

    elif data_source == "Upload Custom Text File (.txt)":
        uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
        if uploaded_file is not None:
            st.session_state.raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"Uploaded file: {uploaded_file.name} ({len(st.session_state.raw_text):,} characters)")

    elif data_source == "Upload CSV File (.csv)":
        uploaded_csv = st.file_uploader("Upload .csv file", type=["csv"])
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write("CSV Preview:", df.head(3))
                text_cols = [c for c in df.columns if df[c].dtype == object]
                if text_cols:
                    selected_col = st.selectbox("Select text column to train on", text_cols)
                    st.session_state.raw_text = " ".join(df[selected_col].dropna().astype(str).tolist())
                    st.success(f"Extracted {len(st.session_state.raw_text):,} characters from column '{selected_col}'")
                else:
                    st.error("No text-based column found in CSV file.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Dataset Validation and Preview
    if st.session_state.raw_text:
        text = st.session_state.raw_text
        st.markdown("---")
        st.subheader("📊 Dataset Statistics & Vocabulary")
        
        c1, c2, c3, c4 = st.columns(4)
        unique_chars = sorted(list(set(text)))
        c1.metric("Total Characters", f"{len(text):,}")
        c2.metric("Unique Vocabulary", f"{len(unique_chars)}")
        c3.metric("Word Count", f"{len(text.split()):,}")
        c4.metric("Line Count", f"{len(text.splitlines()):,}")

        st.text_area("Cleaned Text Preview", text[:500] + ("..." if len(text) > 500 else ""), height=150)

        # Build vocabulary mappings
        char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
        idx_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}
        st.session_state.char_to_idx = char_to_idx
        st.session_state.idx_to_char = idx_to_char

        with st.expander("🔍 View Complete Character Vocabulary"):
            vocab_df = pd.DataFrame([
                {"Index": i, "Character": repr(c), "Unicode": ord(c), "Count": text.count(c)}
                for i, c in enumerate(unique_chars)
            ])
            st.dataframe(vocab_df, use_container_width=True)
    else:
        st.warning("Please upload or select a text corpus above.")

# -------------------------------------------------------------
# 3. Model Architecture & Training
# -------------------------------------------------------------
elif app_mode == "3. Model Architecture & Training":
    st.title("🏋️ Model Architecture & Training")

    if not st.session_state.raw_text or not st.session_state.char_to_idx:
        st.warning("⚠️ Please select or upload a dataset in '2. Dataset & Preprocessing' first.")
    else:
        text = st.session_state.raw_text
        char_to_idx = st.session_state.char_to_idx
        idx_to_char = st.session_state.idx_to_char
        vocab_size = len(char_to_idx)

        st.subheader("1. Configure Hyperparameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            rnn_type = st.selectbox("Recurrent Cell Type", ["LSTM", "SimpleRNN"])
            hidden_units = st.select_slider("Hidden Units", options=[32, 64, 128, 256], value=64)
        with c2:
            seq_length = st.slider("Sequence Length (Window)", min_value=10, max_value=60, value=25)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        with c3:
            epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20)
            learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001, 0.0005], value=0.005)

        # Sequence creation preview
        step = 2
        sentences = [text[i: i + seq_length] for i in range(0, len(text) - seq_length, step)]
        next_chars = [text[i + seq_length] for i in range(0, len(text) - seq_length, step)]
        
        st.info(f"Generated **{len(sentences):,}** training sequence pairs (Input length: {seq_length}, Step: {step}).")

        if st.button("🚀 Start Model Training", type="primary"):
            if len(sentences) < 5:
                st.error("Text is too short to generate sequences. Provide a longer text file.")
            else:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
                from tensorflow.keras.optimizers import Adam

                # Vectorize sequences
                with st.spinner("Vectorizing sequence pairs into one-hot tensors..."):
                    X = np.zeros((len(sentences), seq_length, vocab_size), dtype=np.bool_)
                    y = np.zeros((len(sentences), vocab_size), dtype=np.bool_)
                    for i, sentence in enumerate(sentences):
                        for t, char in enumerate(sentence):
                            X[i, t, char_to_idx[char]] = 1
                        y[i, char_to_idx[next_chars[i]]] = 1

                # Build Model
                Cell = LSTM if rnn_type == "LSTM" else SimpleRNN
                model = Sequential([
                    Cell(hidden_units, input_shape=(seq_length, vocab_size)),
                    Dropout(0.2),
                    Dense(vocab_size, activation="softmax")
                ])
                model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

                progress_bar = st.progress(0)
                status_text = st.empty()
                chart_placeholder = st.empty()

                losses = []
                accuracies = []

                # Custom Epoch Loop
                for epoch in range(epochs):
                    history = model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)
                    loss = history.history["loss"][0]
                    acc = history.history.get("accuracy", [0])[0]
                    losses.append(loss)
                    accuracies.append(acc)

                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc*100:.1f}%")

                    # Live Chart
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(range(1, epoch + 2), losses, 'b-o', label='Categorical Cross-Entropy Loss')
                    ax.set_title("Training Loss Progression", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)

                st.session_state.model = model
                st.session_state.loss_history = losses
                st.success(f"✅ Training completed successfully! Final loss: {losses[-1]:.4f}")

# -------------------------------------------------------------
# 4. Text Generation
# -------------------------------------------------------------
elif app_mode == "4. Text Generation":
    st.title("🔤 Character-by-Character Text Generation")
    
    if st.session_state.model is None or st.session_state.char_to_idx is None:
        st.warning("⚠️ No trained model in memory. Please train a model in section 3, or load a saved model in section 6.")
    else:
        model = st.session_state.model
        char_to_idx = st.session_state.char_to_idx
        idx_to_char = st.session_state.idx_to_char
        seq_length = model.input_shape[1]
        vocab_size = len(char_to_idx)

        c1, c2 = st.columns([2, 1])
        with c1:
            seed_input = st.text_input("Enter Starting Seed Phrase", value="Handwriting is ")
        with c2:
            gen_length = st.slider("Characters to Generate", min_value=50, max_value=500, value=200, step=25)

        temperature = st.slider("Sampling Temperature (T)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
        st.caption(
            "T < 0.5: Conservative & repetitive | T = 0.7: Balanced & natural | T > 1.0: Creative, high entropy"
        )

        if st.button("✨ Generate Text", type="primary"):
            if len(seed_input) < seq_length:
                sentence = seed_input.rjust(seq_length)
            else:
                sentence = seed_input[-seq_length:]

            generated = seed_input
            stream_placeholder = st.empty()

            for _ in range(gen_length):
                x_pred = np.zeros((1, seq_length, vocab_size))
                for t, char in enumerate(sentence):
                    if char in char_to_idx:
                        x_pred[0, t, char_to_idx[char]] = 1.0

                preds = model.predict(x_pred, verbose=0)[0]
                preds = np.asarray(preds).astype('float64')
                preds = np.log(np.maximum(preds, 1e-12)) / temperature
                exp_preds = np.exp(preds)
                probs = exp_preds / np.sum(exp_preds)
                probas = np.random.multinomial(1, probs, 1)
                next_index = np.argmax(probas)
                next_char = idx_to_char.get(next_index, ' ')

                generated += next_char
                sentence = sentence[1:] + next_char
                stream_placeholder.text_area("Generated Output", generated, height=200)

            st.session_state.generated_text = generated
            st.success("Generation completed!")

            # Download button
            st.download_button(
                label="📥 Download Generated Text (.txt)",
                data=generated,
                file_name="generated_handwritten_text.txt",
                mime="text/plain"
            )

# -------------------------------------------------------------
# 5. Kaggle Dataset Analysis
# -------------------------------------------------------------
elif app_mode == "5. Kaggle Dataset Analysis":
    st.title("🔬 Kaggle Dataset Disambiguation & Analysis")
    st.markdown("### Critical Clarification: Text Generation vs. Character Image Datasets")

    st.error(
        "**Key Architectural Finding:** The Kaggle dataset ('english.csv' and handwritten character images) contains "
        "3,410 isolated character images for 62 classes (0-9, A-Z, a-z). It is an **Image Classification / Recognition** "
        "dataset, NOT a continuous text sequence corpus."
    )

    st.markdown("""
    #### ⚠️ Why Attempting to Train an LSTM on `english.csv` Directly to Output Images Fails:
    In naive notebooks, developers attempt:
    - Input: Single Character Index (e.g. `9` or `a`)
    - Architecture: `Embedding -> LSTM(1024) -> Dense(250000) -> Reshape(500, 500, 1)`
    - Loss: Mean Squared Error (`MSE = 0.037`)
    
    **Why this is fundamentally flawed:**
    1. **MSE Averaging / Blurring:** Predicting $500 \\times 500 = 250,000$ pixels with MSE causes the network to predict the *average pixel intensity* of all 55 samples for that character, yielding blurred gray silhouettes rather than crisp strokes.
    2. **No Sentence Semantics:** `english.csv` has no sentence grammar, words, or transition probabilities between characters. An LSTM cannot learn linguistic syntax from an unordered table of independent character labels.
    3. **Proper ML Pipeline:**
       - **Stage 1 (Text Language Modeling):** Character-level RNN/LSTM trained on a text corpus to produce coherent strings.
       - **Stage 2 (Handwriting Rendering):** Synthetic handwriting stroke engine or character glyph compositor that translates generated text into realistic handwriting.
    """)

# -------------------------------------------------------------
# 6. Model Management & Downloads
# -------------------------------------------------------------
elif app_mode == "6. Model Management & Downloads":
    st.title("💾 Model Management & Artifact Downloads")

    st.subheader("1. Download Training Artifacts")
    if st.session_state.loss_history:
        history_df = pd.DataFrame({
            "Epoch": range(1, len(st.session_state.loss_history) + 1),
            "Loss": st.session_state.loss_history
        })
        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download Training Loss History (CSV)",
            data=csv_buffer.getvalue(),
            file_name="training_loss_history.csv",
            mime="text/csv"
        )
    else:
        st.info("Train a model to enable loss history download.")

    if st.session_state.char_to_idx:
        vocab_json = json.dumps(st.session_state.char_to_idx, indent=2)
        st.download_button(
            label="🔤 Download Vocabulary Mapping (JSON)",
            data=vocab_json,
            file_name="char_to_idx.json",
            mime="application/json"
        )

    st.markdown("---")
    st.subheader("2. Save / Load Model Checkpoints")
    model_save_path = "saved_models/streamlit_model.keras"
    if st.button("Save Current Model to Disk"):
        if st.session_state.model is not None:
            os.makedirs("saved_models", exist_ok=True)
            st.session_state.model.save(model_save_path)
            st.success(f"Model saved to {model_save_path}!")
        else:
            st.error("No model in memory to save.")

    uploaded_model = st.file_uploader("Load Saved Keras Model (.keras)", type=["keras", "h5"])
    if uploaded_model is not None:
        try:
            from tensorflow.keras.models import load_model as keras_load
            with open("temp_model.keras", "wb") as f:
                f.write(uploaded_model.read())
            st.session_state.model = keras_load("temp_model.keras")
            st.success("Successfully loaded model into memory!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
