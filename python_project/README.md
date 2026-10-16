# Handwriting Studio: Character-Level RNN Text Generation
### CODSOFT AI/ML Internship — Task 2

A production-grade, submission-ready Machine Learning project demonstrating sequence modeling and character-by-character text generation using genuine Recurrent Neural Networks (SimpleRNN and LSTM) built with TensorFlow/Keras and Python.

---

## 1. Project Overview & Objectives

In character-level language modeling, a deep neural network processes arbitrary text as a discrete sequence of characters rather than whole words. The primary objectives of this project are:
1. **Sequence Learning:** Learn the syntax, vocabulary, morphology, and punctuation rules directly from raw text sequences.
2. **Next-Character Prediction:** Frame text generation as a multi-class classification task where the model predicts the probability distribution of character $c_{t+1}$ given the previous $L$ characters $[c_1, c_2, \dots, c_L]$.
3. **Controllable Text Generation:** Implement **Temperature-Scaled Softmax Sampling** to control the trade-off between conservative, grammatically strict outputs and creative, varied text generation.
4. **Handwriting Synthesis Disambiguation:** Provide an architectural distinction between continuous text sequence generation and isolated character image classification datasets (such as Kaggle's `english.csv`).

---

## 2. Directory Structure

```
├── dataset.py                      # Data loading, cleaning, vocab creation, sequence generation
├── model.py                        # TensorFlow/Keras SimpleRNN and LSTM architectures
├── train.py                        # Training pipeline with live loss tracking, loss curve plotting
├── generate.py                     # Temperature-based character-by-character generation
├── streamlit_app.py                # Full-featured Streamlit user interface
├── kaggle_handwriting_analysis.py  # Analysis & CNN module for Kaggle english.csv dataset
├── sample_text.txt                 # Default corpus for instant offline testing
├── requirements.txt                # Pinned dependencies
├── saved_models/                   # Exported model weights, loss curves, and vocab mappings
│   ├── vocab.json
│   ├── final_char_rnn_model.keras
│   └── training_loss_curve.png
└── README.md                       # Comprehensive internship documentation
```

---

## 3. Mathematical & Architectural Foundation

### Character Tokenization & Sliding Windows
For a text corpus $T$ with unique characters $V = \{c_0, c_1, \dots, c_{K-1}\}$:
- Each character $c$ is assigned a unique index $i \in [0, K-1]$.
- Input sequences $X \in \mathbb{R}^{N \times L \times K}$ are one-hot encoded vectors of sequence length $L$.
- Target outputs $y \in \mathbb{R}^{N \times K}$ represent the one-hot encoded next character.

### Recurrent Cell Update (LSTM)
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(Candidate Cell State)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State Vector)}$$

### Softmax with Temperature Scaling
$$\text{Logits: } z = W_{hy} h_t + b_y$$
$$P(c_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
- **$T = 0.2$ (Low):** Sharpened probability distribution; highly confident and grammatically conservative.
- **$T = 0.7$ (Balanced):** Natural human-like phrasing and vocabulary richness.
- **$T = 1.2+$ (High):** High entropy; creative but higher likelihood of occasional typos.

---

## 4. Local Setup & Execution Guide

### Step 1: Install Dependencies
Ensure Python 3.9+ is installed, then run:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Character-Level RNN
Train on the bundled sample text or your own uploaded text file:
```bash
# Train using an LSTM with 128 hidden units for 30 epochs:
python train.py --data sample_text.txt --epochs 30 --seq_length 30 --hidden_units 128 --rnn_type LSTM
```
Artifacts will be automatically saved into the `saved_models/` folder:
- `saved_models/final_char_rnn_model.keras`
- `saved_models/vocab.json`
- `saved_models/training_loss_curve.png`
- `saved_models/training_metrics.csv`

### Step 3: Generate Text
Generate new text character by character using your trained model:
```bash
python generate.py --seed "Handwriting is " --length 250 --temperature 0.7 --output generated_result.txt
```

### Step 4: Run the Streamlit Graphical Interface
Launch the interactive web application:
```bash
streamlit run streamlit_app.py
```
This opens the application in your browser at `http://localhost:8501`.

---

## 5. Google Colab Execution Instructions

For zero-setup cloud execution with free GPU acceleration:
1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook and choose **Runtime > Change runtime type > T4 GPU**.
3. Upload the project files (`dataset.py`, `model.py`, `train.py`, `generate.py`, `sample_text.txt`).
4. Execute the following cells:

```python
# Cell 1: Check GPU & Install packages
!nvidia-smi
!pip install -r requirements.txt

# Cell 2: Train the model
!python train.py --data sample_text.txt --epochs 40 --seq_length 35 --hidden_units 128 --rnn_type LSTM

# Cell 3: Display training loss graph
from IPython.display import Image
Image('saved_models/training_loss_curve.png')

# Cell 4: Generate text with multiple temperatures
!python generate.py --seed "Handwriting is " --length 300 --temperature 0.3
!python generate.py --seed "Handwriting is " --length 300 --temperature 0.7
!python generate.py --seed "Handwriting is " --length 300 --temperature 1.2
```

---

## 6. Important Note on Kaggle English Handwritten Characters Dataset

The Kaggle dataset `english-handwritten-characters-dataset` (`english.csv`) consists of **3,410 single character images** corresponding to 62 classes (`0-9`, `A-Z`, `a-z`, 55 samples each).

- **Architectural Clarification:** An image classification dataset with single character labels cannot serve as a text corpus because it contains no sentence transitions, syntax, or word dependencies.
- **The Right Architecture:**
  - **Text Generation (Task 2):** Use the Character-Level RNN pipeline provided here on a text corpus.
  - **Handwriting Image Recognition / Classification:** Use `kaggle_handwriting_analysis.py` which provides a 3-layer Convolutional Neural Network (CNN) specifically tailored for `english.csv`.
  - **Handwriting Synthesis:** Translate the RNN's generated textual output into calligraphic and handwritten document renderings via our synthesis module.

---

## 7. Submission Checklist & Screenshots Guide

Capture the following 8 screenshots for your internship submission report:
1. **Dataset Ingestion & Statistics:** Screenshot of `dataset.py` running or Streamlit dataset preview displaying character vocabulary size.
2. **Vocabulary Mapping Table:** Screenshot showing the `char_to_idx` table and one-hot encoding shape.
3. **Model Summary:** Screenshot of `model.summary()` displaying layer shapes and parameter counts.
4. **Epoch Training Progress:** Screenshot of the terminal/Colab output showing loss decreasing across epochs.
5. **Loss Curve Plot:** Screenshot of `saved_models/training_loss_curve.png` showing smooth convergence.
6. **Temperature Comparison:** Side-by-side screenshot showing generation at $T=0.2$, $T=0.7$, and $T=1.2$.
7. **Streamlit / Web Studio Interface:** Full-screen capture of the running Handwriting Studio application.
8. **Artifact Directory:** Screenshot showing `saved_models/` containing the `.keras` model, `vocab.json`, and metrics CSV.
