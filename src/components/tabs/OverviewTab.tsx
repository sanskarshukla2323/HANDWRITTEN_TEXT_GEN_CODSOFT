import React from 'react';
import { ArrowRight, Brain, Cpu, Database, FileText, Layers, Play, Sparkles } from 'lucide-react';
import { TabType } from '../Header';

interface OverviewTabProps {
  onNavigate: (tab: TabType) => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-10">
      
      {/* Hero Section */}
      <div className="relative rounded-2xl bg-gradient-to-b from-slate-800/80 to-slate-900 border border-slate-700/60 p-6 sm:p-8 overflow-hidden">
        <div className="max-w-3xl space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400">
            <span>CODSOFT AI/ML INTERNSHIP</span>
            <span aria-hidden="true">·</span>
            <span>TASK 2 SUBMISSION</span>
            <span aria-hidden="true">·</span>
            <span>RECURRENT NEURAL NETWORKS</span>
          </div>
          
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-white" style={{ textWrap: 'balance' }}>
            Handwritten Text Generation using Character-Level RNN
          </h1>
          
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            A complete, production-grade deep learning system that learns character transition probabilities directly from text corpora to generate natural sequences one character at a time. Accompanied by a realistic handwriting stroke synthesis module and thorough disambiguation of the Kaggle handwritten characters dataset.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => onNavigate('dataset')}
              className="flex items-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer"
            >
              <span>Explore Dataset & Vocab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('training')}
              className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
            >
              <span>Train Model Live</span>
            </button>
            <button
              onClick={() => onNavigate('handwriting')}
              className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-indigo-900/40 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
            >
              <span>Handwriting Canvas</span>
            </button>
          </div>
        </div>

        {/* Technical Highlights Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-8 pt-6 border-t border-slate-800/80 text-xs">
          <div>
            <span className="text-slate-400 block mb-0.5">Model Architecture</span>
            <span className="text-slate-100 font-semibold font-mono-code">SimpleRNN & LSTM</span>
          </div>
          <div>
            <span className="text-slate-400 block mb-0.5">Input Granularity</span>
            <span className="text-slate-100 font-semibold font-mono-code">Character-Level (V ~ 60-90)</span>
          </div>
          <div>
            <span className="text-slate-400 block mb-0.5">Sampling Control</span>
            <span className="text-slate-100 font-semibold font-mono-code">Temperature Softmax</span>
          </div>
          <div>
            <span className="text-slate-400 block mb-0.5">Training Objective</span>
            <span className="text-slate-100 font-semibold font-mono-code">Categorical Cross-Entropy</span>
          </div>
        </div>
      </div>

      {/* Core Workflow Pipeline */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">01. End-to-End Deep Learning Pipeline</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-3">
              1
            </div>
            <h3 className="text-sm font-semibold text-white">Character Ingestion</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Text is sanitized and tokenized at character resolution. Unique symbols form vocabulary $V$. Bidirectional lookup tables (<code className="text-indigo-300">char_to_idx</code> and <code className="text-indigo-300">idx_to_char</code>) are constructed.
            </p>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-3">
              2
            </div>
            <h3 className="text-sm font-semibold text-white">Sliding Window ($X \to y$)</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Sequences of fixed length $L$ (e.g. 25–40 characters) are sliced with stride step $S$. Input tensor $X$ contains the sequence, and target $y$ is the single next character symbol.
            </p>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-3">
              3
            </div>
            <h3 className="text-sm font-semibold text-white">Recurrent State Update</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              At step t, recurrent cell integrates input x_t with previous memory h_(t-1). In LSTMs, input, forget, and output gates prevent gradient vanishing across long textual contexts.
            </p>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-xs mb-3">
              4
            </div>
            <h3 className="text-sm font-semibold text-white">Temperature Generation</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Logits are scaled by temperature $T$. Softmax converts logits into probabilities. The network samples the next character, appends it to output, and feeds it forward iteratively.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Mathematical Formulations</h3>
          </div>

          <div className="space-y-3 text-xs text-slate-300">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 font-mono-code space-y-1">
              <span className="text-slate-400 block">// SimpleRNN Hidden State:</span>
              <span className="text-indigo-300">h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)</span>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 font-mono-code space-y-1">
              <span className="text-slate-400 block">// Softmax Output with Temperature T:</span>
              <span className="text-indigo-300">P(c_i | x_1...x_t) = exp(z_i / T) / Σ_j exp(z_j / T)</span>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 font-mono-code space-y-1">
              <span className="text-slate-400 block">// Categorical Cross-Entropy Loss:</span>
              <span className="text-indigo-300">Loss = - (1/N) Σ log P(y_target)</span>
              <span className="text-slate-400 block">// Perplexity:</span>
              <span className="text-indigo-300">Perplexity = exp(Loss)</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-white">Character-Level vs. Word-Level RNN</h3>
          </div>

          <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="font-semibold text-indigo-300 block mb-1">Zero Out-of-Vocabulary (OOV) Rate:</span>
              A character vocabulary is tiny (typically 60–90 symbols including punctuation and spaces), completely eliminating the out-of-vocabulary words problem common in word-level tokenization.
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="font-semibold text-indigo-300 block mb-1">Learns Sub-Word Morphology:</span>
              The model naturally learns prefixes, suffixes, spelling conventions, hyphenation, and grammatical structure from scratch.
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="font-semibold text-indigo-300 block mb-1">Direct Handwriting Translation:</span>
              Since human handwriting consists of discrete character strokes and cursive ligatures, character-level sequence modeling directly mirrors physical pen movements.
            </div>
          </div>
        </div>
      </div>

      {/* Section 4 Clarification Callout */}
      <div className="rounded-xl bg-slate-800/60 border border-indigo-900/50 p-6 space-y-3">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold">
          <Database className="w-4 h-4" />
          <span>IMPORTANT ARCHITECTURAL CLARIFICATION (SECTION 4 OF CODSOFT SPECIFICATION)</span>
        </div>
        <h3 className="text-base font-semibold text-white">
          Character-Level RNN Text Generation vs. Kaggle Handwritten Images
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          The Kaggle <em>English Handwritten Characters Dataset</em> (<code className="text-indigo-300">english.csv</code>) contains 3,410 isolated character images for 62 classes (0-9, A-Z, a-z). 
          Attempting to train an LSTM to directly predict 500x500 image pixels from single characters results in blurry pixel averaging (MSE ~0.037) and completely lacks sentence-level linguistic semantics.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          In this project, we implement the <strong>proper two-stage machine learning architecture</strong>:
          <br />
          <strong>Stage 1 (Text Sequence Modeling):</strong> The Character-Level RNN generates coherent, novel text sequences.
          <br />
          <strong>Stage 2 (Handwriting Rendering & Kaggle Analysis):</strong> The generated text is passed to our Handwriting Canvas engine for realistic calligraphic rendering, while a separate CNN module handles Kaggle character classification.
        </p>
        <div className="pt-2">
          <button
            onClick={() => onNavigate('kaggle')}
            className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1.5 cursor-pointer"
          >
            <span>Read full Kaggle english.csv technical breakdown</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

    </div>
  );
};
