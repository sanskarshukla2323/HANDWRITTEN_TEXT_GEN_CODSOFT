import React, { useState } from 'react';
import { CheckCircle2, Camera, HelpCircle, FileCheck, Copy, Check, Award, BookOpen } from 'lucide-react';

export const SubmissionGuideTab: React.FC = () => {
  const [copiedSection, setCopiedSection] = useState<string | null>(null);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(id);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const screenshots = [
    {
      id: 'ss1',
      title: '1. Dataset Ingestion & Preprocessing',
      instructions: 'Capture the Dataset tab showing total characters, unique vocabulary size (~60-90 symbols), and the active text preview.',
      where: 'Dataset Tab or Colab: print(f"Vocab size: {len(chars)}")',
    },
    {
      id: 'ss2',
      title: '2. Character Vocabulary Mapping Table',
      instructions: 'Capture the char_to_idx mapping table showing characters, decimal Unicode codes, and frequency counts.',
      where: 'Dataset Tab -> Vocabulary Mapping Table or vocab.json',
    },
    {
      id: 'ss3',
      title: '3. Model Architecture Summary',
      instructions: 'Capture model.summary() displaying the LSTM/SimpleRNN layer, Dropout, and Dense output layer with parameter counts.',
      where: 'Training Tab or terminal output from python model.py',
    },
    {
      id: 'ss4',
      title: '4. Epoch Training Progression',
      instructions: 'Capture terminal/Colab log showing cross-entropy loss decreasing epoch by epoch (e.g. from 3.4 down to < 1.5).',
      where: 'Terminal / Colab execution of python train.py',
    },
    {
      id: 'ss5',
      title: '5. Training Loss Curve Graph',
      instructions: 'Capture the Matplotlib loss graph (training_loss_curve.png) displaying smooth convergence.',
      where: 'Training Tab SVG chart or saved_models/training_loss_curve.png',
    },
    {
      id: 'ss6',
      title: '6. Temperature Generation Comparison',
      instructions: 'Capture text generation outputs comparing T=0.2 (conservative) vs T=0.7 (natural) vs T=1.2 (creative).',
      where: 'Generation Tab or Colab execution of python generate.py',
    },
    {
      id: 'ss7',
      title: '7. Handwriting Canvas Document Synthesis',
      instructions: 'Capture the generated text rendered in calligraphy cursive on lined notebook paper.',
      where: 'Handwriting Canvas Tab',
    },
    {
      id: 'ss8',
      title: '8. Kaggle english.csv Disambiguation Analysis',
      instructions: 'Capture the Kaggle Dataset tab explaining why character image classification differs from text language modeling.',
      where: 'Kaggle Dataset Tab',
    },
  ];

  const vivaQuestions = [
    {
      q: 'Why choose Character-Level over Word-Level modeling for this task?',
      a: 'Character-level modeling has zero Out-of-Vocabulary (OOV) tokens, handles punctuation and spelling variations naturally, requires a tiny vocabulary (60–90 symbols vs tens of thousands of words), and directly matches the stroke-by-stroke nature of handwritten characters.',
    },
    {
      q: 'What is the role of Temperature (T) in softmax sampling?',
      a: 'Temperature scales the output logits prior to exponentiation: P(c_i) = exp(z_i / T) / sum_j exp(z_j / T). When T < 1, high-probability tokens are amplified, making output conservative and grammatically strict. When T > 1, the probability distribution is flattened, encouraging exploratory and diverse text generation.',
    },
    {
      q: 'How does an LSTM resolve the vanishing gradient problem in RNNs?',
      a: 'LSTMs introduce an additive cell state (C_t) regulated by forget, input, and output gates. Unlike standard SimpleRNNs where gradients vanish through continuous matrix multiplications, error signals flow through the linear cell state without exponential decay across time steps.',
    },
    {
      q: 'Why can\'t we train an LSTM directly on Kaggle english.csv to output images?',
      a: 'english.csv contains 3,410 isolated character images without linguistic sequence structure. Training an LSTM to predict 250,000 raw pixels using MSE loss averages all sample variations, resulting in blurred gray blobs. The correct pipeline uses a character-level RNN for sequence text generation, and a handwriting stroke engine or CNN for image rendering.',
    },
  ];

  const linkedinTemplate = `🚀 Excited to share my completion of Task 2 for the CODSOFT AI/ML Internship!

📌 Project Title: Handwritten Text Generation using Character-Level RNN (Handwriting Studio)
🔍 Tech Stack: Python, TensorFlow/Keras, LSTM, SimpleRNN, NumPy, Streamlit, React

🎯 Highlights:
• Built a genuine character-level Recurrent Neural Network (RNN/LSTM) trained to predict sequence probabilities.
• Implemented Temperature-Scaled Softmax Sampling (T = 0.2 to 1.5) to balance diversity vs. grammar.
• Built a full-featured web interface ("Handwriting Studio") featuring real-time training loss curves, sliding window sequence inspector, and a handwriting canvas synthesis engine.
• Conducted an architectural analysis of the Kaggle English Handwritten Characters Dataset (english.csv), comparing sequential language models against isolated character image classification.

Special thanks to CODSOFT for the guidance and opportunity!

#MachineLearning #DeepLearning #CODSOFT #ArtificialIntelligence #Python #TensorFlow #NLP #Internship`;

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">08. CODSOFT Internship Submission & Viva Guide</h2>
        <p className="text-xs text-slate-400 mt-1">
          Everything you need to successfully submit Task 2: exact screenshot checklist, presentation summary, and viva Q&A answers.
        </p>
      </div>

      {/* Submission Checklist */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Camera className="w-4 h-4 text-indigo-400" />
          <span>Mandatory 8-Screenshot Checklist for Your Submission</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          {screenshots.map((ss) => (
            <div key={ss.id} className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1.5">
              <span className="font-bold text-indigo-300 block">{ss.title}</span>
              <p className="text-slate-300 text-[11px] leading-relaxed">{ss.instructions}</p>
              <span className="text-[10px] text-slate-500 font-mono-code block pt-0.5">Location: {ss.where}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Viva / Interview Questions & Answers */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-amber-400" />
          <span>Viva Voce & Technical Defense Q&A Cheat Sheet</span>
        </h3>

        <div className="space-y-3">
          {vivaQuestions.map((vq, i) => (
            <div key={i} className="p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-2 text-xs">
              <span className="font-bold text-amber-300 block">Q{i + 1}: {vq.q}</span>
              <p className="text-slate-300 leading-relaxed pl-2 border-l-2 border-indigo-500/50">
                {vq.a}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* LinkedIn & GitHub Submission Template */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-400" />
            <span>Ready-to-Use LinkedIn / Social Submission Post</span>
          </h3>
          <button
            onClick={() => handleCopy(linkedinTemplate, 'linkedin')}
            className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-xs transition-colors cursor-pointer"
          >
            {copiedSection === 'linkedin' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedSection === 'linkedin' ? 'Copied Post' : 'Copy Post'}</span>
          </button>
        </div>

        <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono-code text-xs text-slate-300 whitespace-pre-wrap leading-relaxed max-h-56 overflow-y-auto">
          {linkedinTemplate}
        </pre>
      </div>

    </div>
  );
};
