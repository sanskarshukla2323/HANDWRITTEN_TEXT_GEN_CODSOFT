import React, { useState } from 'react';
import { Database, AlertTriangle, CheckCircle, Code, Layers, Info, ExternalLink } from 'lucide-react';
import { KAGGLE_ENGLISH_CSV_EXPLAINER } from '../../data/sampleDatasets';

export const KaggleAnalysisTab: React.FC = () => {
  const [selectedClass, setSelectedClass] = useState<string>('A');

  const digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  const lowercase = 'abcdefghijklmnopqrstuvwxyz'.split('');

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">06. Kaggle Dataset Analysis & Disambiguation</h2>
        <p className="text-xs text-slate-400 mt-1">
          Deep technical breakdown of Kaggle's <em>English Handwritten Characters Dataset</em> (<code className="text-indigo-300">english.csv</code>) and why text language models must be separated from isolated character image classification.
        </p>
      </div>

      {/* Primary Alert Callout */}
      <div className="p-5 rounded-xl bg-amber-950/40 border border-amber-800/60 space-y-2">
        <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>Section 4 Clarification: Character-Level RNN Text Generation vs. Handwritten Images</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          Character-level RNN text generation produces <strong>textual sequences</strong>, not raw bitmap images. 
          Kaggle's <code className="text-amber-300">english.csv</code> dataset contains isolated single-character images paired with one-letter labels (e.g. <code className="text-slate-300">Img/img010-001.png, 9</code>). 
          It contains no sentences, grammatical transitions, or sequential linguistic context.
        </p>
      </div>

      {/* Comparative Matrix: Language Model vs Character Image Dataset */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm">
            <CheckCircle className="w-4 h-4" />
            <span>Character-Level RNN (CODSOFT Task 2)</span>
          </div>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-indigo-400 font-bold">•</span>
              <span><strong>Problem Type:</strong> Natural Language Sequence Modeling (Autoregressive Next-Token Prediction).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-400 font-bold">•</span>
              <span><strong>Input Data:</strong> Sequential sentences or literature passages (e.g. essay, Alice in Wonderland).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-400 font-bold">•</span>
              <span><strong>Loss Function:</strong> Categorical Cross-Entropy across vocabulary $V$.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-indigo-400 font-bold">•</span>
              <span><strong>Outcome:</strong> Generates coherent textual prose, novel thoughts, and grammatically structured sentences.</span>
            </li>
          </ul>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
            <Info className="w-4 h-4" />
            <span>Kaggle English Handwritten Dataset (english.csv)</span>
          </div>
          <ul className="space-y-2 text-xs text-slate-300">
            <li className="flex items-start gap-2">
              <span className="text-amber-400 font-bold">•</span>
              <span><strong>Problem Type:</strong> Isolated Character Image Recognition / Optical Character Recognition (OCR).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-400 font-bold">•</span>
              <span><strong>Input Data:</strong> 3,410 single character images across 62 discrete classes (0-9, A-Z, a-z).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-400 font-bold">•</span>
              <span><strong>Optimal Model:</strong> 2D Convolutional Neural Network (CNN) or ResNet classifier.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-400 font-bold">•</span>
              <span><strong>Why LSTM Image Regression Fails:</strong> Predicting 250,000 pixels with MSE averages all 55 samples for a character, causing blurry ghost images.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Dataset Structure Explorer */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-indigo-400" />
              <span>Kaggle english.csv Structure & 62-Class Matrix</span>
            </h3>
            <p className="text-xs text-slate-400">
              3,410 total images (55 handwritten samples per character class).
            </p>
          </div>
          <a
            href={KAGGLE_ENGLISH_CSV_EXPLAINER.kaggleUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-semibold"
          >
            <span>View Kaggle Dataset</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* Character Class Selectors */}
        <div className="space-y-3 pt-2 text-xs">
          <div>
            <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">Digits (0-9, img001 to img010):</span>
            <div className="flex flex-wrap gap-1.5">
              {digits.map((d) => (
                <button
                  key={d}
                  onClick={() => setSelectedClass(d)}
                  className={`w-7 h-7 rounded font-mono-code font-bold text-xs transition-colors cursor-pointer ${
                    selectedClass === d
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">Uppercase (A-Z, img011 to img036):</span>
            <div className="flex flex-wrap gap-1.5">
              {uppercase.map((c) => (
                <button
                  key={c}
                  onClick={() => setSelectedClass(c)}
                  className={`w-7 h-7 rounded font-mono-code font-bold text-xs transition-colors cursor-pointer ${
                    selectedClass === c
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">Lowercase (a-z, img037 to img062):</span>
            <div className="flex flex-wrap gap-1.5">
              {lowercase.map((c) => (
                <button
                  key={c}
                  onClick={() => setSelectedClass(c)}
                  className={`w-7 h-7 rounded font-mono-code font-bold text-xs transition-colors cursor-pointer ${
                    selectedClass === c
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Selected Class Inspector */}
        <div className="p-4 rounded-lg bg-slate-900 border border-slate-700/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center font-handwriting-apple text-3xl text-indigo-300 shadow-inner">
              {selectedClass}
            </div>
            <div className="space-y-0.5 text-xs">
              <span className="text-white font-bold block">Selected Character Class: '{selectedClass}'</span>
              <span className="text-slate-400 font-mono-code block">Sample Count: 55 handwritten images</span>
              <span className="text-slate-400 font-mono-code block">ASCII Code: {selectedClass.charCodeAt(0)}</span>
            </div>
          </div>

          <div className="text-xs text-slate-400 max-w-sm">
            In our complete architecture, this character class is mapped to stroke vectors in the <span className="text-indigo-300 font-semibold">Handwriting Canvas</span>, while the Character-Level RNN handles sequential language generation.
          </div>
        </div>
      </div>

      {/* Recommended CNN Python Architecture for english.csv */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Code className="w-4 h-4 text-indigo-400" />
          <span>Recommended TensorFlow CNN Architecture for english.csv</span>
        </h3>
        <p className="text-xs text-slate-400">
          If you choose to train a model specifically on the image files of <code className="text-indigo-300">english.csv</code>, use the Convolutional Neural Network below instead of an LSTM image regressor:
        </p>

        <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono-code text-xs text-slate-200 overflow-x-auto leading-relaxed">
{`from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# 3-Layer Convolutional Neural Network for 62 Character Classes
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
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
    Dense(62, activation='softmax')  # 62 classes (0-9, A-Z, a-z)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`}
        </pre>
      </div>

    </div>
  );
};
