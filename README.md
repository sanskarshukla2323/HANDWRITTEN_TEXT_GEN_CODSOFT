# Handwriting Studio: Character-Level RNN Text Generation
### CODSOFT AI/ML Internship — Task 2

A complete, beginner-friendly, submission-ready project for **CODSOFT AI/ML Internship Task 2: Handwritten Text Generation using Character-Level RNN**.

This repository includes:
1. **Interactive Web Application ("Handwriting Studio"):** Built with React, TypeScript, Tailwind CSS, and Lucide icons. Runs a genuine in-browser Character-Level RNN neural network engine with real-time BPTT training, loss curves, temperature sampling, and handwriting canvas simulation.
2. **Complete Python & TensorFlow Project (`/python_project`):** Standalone, runnable Python scripts (`train.py`, `generate.py`, `model.py`, `dataset.py`, `streamlit_app.py`, `requirements.txt`) ready for local execution and Google Colab with GPU acceleration.
3. **Kaggle English Handwritten Characters Dataset Analysis:** Clarification and CNN implementation for `english.csv`.
4. **Export & Submission Suite:** 1-click ZIP export containing all code, datasets, and guides, complete with a CODSOFT submission checklist and screenshot guide.

---

## Quick Start: Running the Interactive Web Studio
```bash
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Quick Start: Running the Python Project
```bash
cd python_project
pip install -r requirements.txt

# Train the model
python train.py --data sample_text.txt --epochs 30 --hidden_units 128 --rnn_type LSTM

# Generate text
python generate.py --seed "Handwriting is " --length 200 --temperature 0.7

# Run Streamlit Web App
streamlit run streamlit_app.py
```

---

## Full Documentation
For complete technical details, mathematical derivations, Colab notebook instructions, and screenshot guidelines, see [python_project/README.md](python_project/README.md).
