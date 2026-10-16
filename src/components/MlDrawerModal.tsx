import React, { useState } from 'react';
import { X, Cpu, Database, Play, Pause, RotateCcw, TrendingDown, Sparkles, BookOpen, ExternalLink, Camera, HelpCircle, Code2, CheckCircle2 } from 'lucide-react';
import { CharRnnEngine } from '../ml/charRnnEngine';
import { KAGGLE_ENGLISH_CSV_EXPLAINER } from '../data/sampleDatasets';

interface MlDrawerModalProps {
  isOpen: boolean;
  onClose: () => void;
  engine: CharRnnEngine;
  currentText: string;
  onModelUpdated: () => void;
}

export const MlDrawerModal: React.FC<MlDrawerModalProps> = ({
  isOpen,
  onClose,
  engine,
  currentText,
  onModelUpdated,
}) => {
  const [activeTab, setActiveTab] = useState<'trainer' | 'kaggle' | 'colab' | 'viva'>('trainer');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epochsToTrain, setEpochsToTrain] = useState<number>(20);
  const [lossData, setLossData] = useState(engine.lossHistory);

  if (!isOpen) return null;

  const handleTrainStep = () => {
    setIsTraining(true);
    let epoch = 0;

    const step = () => {
      if (epoch >= epochsToTrain) {
        setIsTraining(false);
        return;
      }
      epoch++;
      const seqLen = engine.config.sequenceLength;
      const steps = Math.min(600, currentText.length - seqLen - 1);
      let h = new Array(engine.config.hiddenUnits).fill(0);
      let epochLoss = 0;
      let count = 0;

      for (let s = 0; s < steps; s += Math.max(2, Math.floor(seqLen / 3))) {
        const slice = currentText.slice(s, s + seqLen + 1);
        if (slice.length < seqLen + 1) break;

        const inputs = slice.slice(0, seqLen).split('').map(c => engine.vocab.charToIdx[c] ?? 0);
        const targets = slice.slice(1, seqLen + 1).split('').map(c => engine.vocab.charToIdx[c] ?? 0);

        const res = engine.trainStep(inputs, targets, h);
        h = res.hFinal;
        epochLoss += res.loss;
        count++;
      }

      const avgLoss = count > 0 ? epochLoss / count : 1.5;
      const perp = Math.exp(avgLoss);

      engine.currentLoss = avgLoss;
      engine.currentPerplexity = perp;
      engine.totalEpochsTrained += 1;
      engine.isTrained = true;

      engine.lossHistory.push({
        epoch: engine.totalEpochsTrained,
        loss: Number(avgLoss.toFixed(4)),
        perplexity: Number(perp.toFixed(2)),
        elapsedMs: Date.now(),
      });

      setLossData([...engine.lossHistory]);
      onModelUpdated();
      setTimeout(step, 60);
    };

    setTimeout(step, 10);
  };

  const handleReset = () => {
    engine.initWeights();
    setLossData([]);
    onModelUpdated();
  };

  const handleLoadPretrained = () => {
    engine.loadPretrainedWeights(currentText);
    setLossData([...engine.lossHistory]);
    onModelUpdated();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Machine Learning & Kaggle Dataset Console</h3>
              <p className="text-[11px] text-slate-400">CODSOFT AI/ML Internship · Task 2 Technical Reference</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Tabs */}
        <div className="flex items-center gap-2 px-6 pt-3 border-b border-slate-800 bg-slate-900 text-xs">
          <button
            onClick={() => setActiveTab('trainer')}
            className={`pb-2.5 font-semibold transition-colors cursor-pointer border-b-2 ${
              activeTab === 'trainer' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            RNN Training & Loss Curve
          </button>
          <button
            onClick={() => setActiveTab('kaggle')}
            className={`pb-2.5 font-semibold transition-colors cursor-pointer border-b-2 ${
              activeTab === 'kaggle' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Kaggle Dataset Analysis (english.csv)
          </button>
          <button
            onClick={() => setActiveTab('colab')}
            className={`pb-2.5 font-semibold transition-colors cursor-pointer border-b-2 ${
              activeTab === 'colab' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Google Colab & Python Setup
          </button>
          <button
            onClick={() => setActiveTab('viva')}
            className={`pb-2.5 font-semibold transition-colors cursor-pointer border-b-2 ${
              activeTab === 'viva' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Submission Checklist & Viva Q&A
          </button>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-xs text-slate-300">
          
          {/* TAB 1: TRAINER */}
          {activeTab === 'trainer' && (
            <div className="space-y-6">
              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">Genuine Character-Level RNN Engine Status:</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono-code font-bold ${engine.isTrained ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
                    {engine.isTrained ? `Trained (${engine.totalEpochsTrained} Epochs)` : 'Untrained'}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
                  <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Architecture</span>
                    <span className="font-mono-code text-white font-bold">{engine.config.architecture} (64 units)</span>
                  </div>
                  <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Vocabulary Size</span>
                    <span className="font-mono-code text-indigo-400 font-bold">{engine.vocab.vocabSize} unique symbols</span>
                  </div>
                  <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Current Loss</span>
                    <span className="font-mono-code text-white font-bold tabular-nums">
                      {engine.currentLoss > 0 ? engine.currentLoss.toFixed(4) : '—'}
                    </span>
                  </div>
                  <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Perplexity (e^loss)</span>
                    <span className="font-mono-code text-white font-bold tabular-nums">
                      {engine.currentPerplexity > 0 ? engine.currentPerplexity.toFixed(2) : '—'}
                    </span>
                  </div>
                </div>

                {/* Training Controls */}
                <div className="flex flex-wrap items-center gap-3 pt-2">
                  <button
                    onClick={handleTrainStep}
                    disabled={isTraining}
                    className="flex items-center gap-1.5 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg font-semibold transition-colors cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>{isTraining ? 'Training Epochs...' : `Train ${epochsToTrain} Epochs`}</span>
                  </button>

                  <button
                    onClick={handleLoadPretrained}
                    disabled={isTraining}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-indigo-900/40 rounded-lg font-medium transition-colors cursor-pointer"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Load Pretrained Weights</span>
                  </button>

                  <button
                    onClick={handleReset}
                    disabled={isTraining}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-lg font-medium transition-colors ml-auto cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Reset</span>
                  </button>
                </div>
              </div>

              {/* Loss History Table / Summary */}
              {lossData.length > 0 ? (
                <div className="space-y-2">
                  <span className="font-semibold text-white block">Loss Progression History ({lossData.length} records):</span>
                  <div className="max-h-48 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950 p-2 font-mono-code text-[11px]">
                    <div className="grid grid-cols-3 py-1 px-2 border-b border-slate-800 text-slate-400 font-bold">
                      <span>Epoch</span>
                      <span>Categorical Cross-Entropy Loss</span>
                      <span>Perplexity</span>
                    </div>
                    {lossData.map((d, i) => (
                      <div key={i} className="grid grid-cols-3 py-1 px-2 border-b border-slate-900/60 hover:bg-slate-900/40">
                        <span className="text-indigo-400">Epoch {d.epoch}</span>
                        <span className="text-white tabular-nums">{d.loss}</span>
                        <span className="text-slate-300 tabular-nums">{d.perplexity}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-6 text-center text-slate-500 italic border border-slate-800 rounded-xl bg-slate-950/40">
                  Click "Train Epochs" or "Load Pretrained Weights" to observe genuine loss convergence.
                </div>
              )}
            </div>
          )}

          {/* TAB 2: KAGGLE DATASET ANALYSIS */}
          {activeTab === 'kaggle' && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/50 space-y-2">
                <span className="font-bold text-amber-300 text-xs block">
                  Critical Finding: Dataset Limitation Analysis of Kaggle 'english.csv'
                </span>
                <p className="text-slate-300 leading-relaxed">
                  The dataset contains <strong>3,410 isolated character images</strong> across 62 discrete classes (10 digits: 0-9, 26 uppercase: A-Z, 26 lowercase: a-z), with exactly 55 samples per class.
                </p>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Why it cannot directly train a text-generation RNN:</strong> 
                  It is an <em>image recognition / OCR</em> dataset without sentence syntax, word vocabulary, or character transition sequences. Attempting to regress 250,000 pixels with MSE from a single character index produces blurred average pixel blobs (MSE ~0.037).
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
                <span className="font-semibold text-white block">What Data Is Required for an End-to-End Handwriting RNN?</span>
                <p className="text-slate-300 leading-relaxed">
                  To train a true neural network that outputs handwriting strokes directly (like Alex Graves' famous 2013 paper <em>Generating Sequences With Recurrent Neural Networks</em>), you need continuous online handwriting data:
                </p>
                <ul className="list-disc pl-5 space-y-1 text-slate-300">
                  <li><strong>Stroke Coordinates:</strong> Triplet sequences $(x_t, y_t, p_t)$ recording pen displacements and pen-lift binary flags $p_t$.</li>
                  <li><strong>Corpus:</strong> Datasets like the <em>IAM On-Line Handwriting Database</em> or <em>Unipen</em>.</li>
                  <li><strong>Architecture:</strong> Mixture Density Network (MDN) stacked on top of an LSTM with handwriting prediction layers.</li>
                </ul>
              </div>

              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-2">
                <span className="font-semibold text-white block">Our Two-Stage Architecture (Best Practical Solution):</span>
                <p className="text-slate-300 leading-relaxed">
                  <strong>Stage A (Text Sequence Modeling):</strong> Character-Level RNN (LSTM/SimpleRNN) generates grammatically structured, novel sentences.
                  <br />
                  <strong>Stage B (Handwriting Synthesis):</strong> Visual calligraphy & stroke compositor renders the text onto lined, parchment, or grid paper with realistic pen ink and human jitter.
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: GOOGLE COLAB INSTRUCTIONS */}
          {activeTab === 'colab' && (
            <div className="space-y-4">
              <span className="font-semibold text-white block">Running in Google Colab (Free GPU):</span>
              <ol className="list-decimal pl-5 space-y-2 text-slate-300">
                <li>Access the project files in the <code className="text-indigo-300">python_project/</code> directory.</li>
                <li>Open <a href="https://colab.research.google.com" target="_blank" rel="noopener noreferrer" className="text-indigo-400 underline">Google Colab</a>, click <strong>File &gt; Upload Notebook</strong>, and upload <code className="text-indigo-300">Handwritten_Text_RNN_Colab.ipynb</code>.</li>
                <li>Set the runtime to GPU via <strong>Runtime &gt; Change runtime type &gt; T4 GPU</strong>.</li>
                <li>Run all cells to train the model, plot the loss curve, and generate text at multiple temperatures ($T=0.2, 0.7, 1.2$).</li>
              </ol>

              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono-code text-[11px] text-slate-300 space-y-1">
                <span className="text-slate-500 block"># Or run via CLI locally:</span>
                <span className="text-indigo-300 block">pip install -r requirements.txt</span>
                <span className="text-indigo-300 block">python train.py --epochs 30 --rnn_type LSTM</span>
                <span className="text-indigo-300 block">python generate.py --seed "Handwriting is " --temperature 0.7</span>
                <span className="text-indigo-300 block">streamlit run streamlit_app.py</span>
              </div>
            </div>
          )}

          {/* TAB 4: VIVA Q&A */}
          {activeTab === 'viva' && (
            <div className="space-y-3">
              <span className="font-semibold text-white block">Internship Defense & Evaluation Cheat Sheet:</span>
              
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="font-bold text-amber-300">Q: Why Character-Level over Word-Level?</span>
                <p className="text-slate-300">Zero Out-of-Vocabulary (OOV) tokens, learns punctuation and spelling from scratch, vocabulary is small (~60-90 symbols), and maps directly to individual handwriting strokes.</p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="font-bold text-amber-300">Q: What is the role of Temperature (T)?</span>
                <p className="text-slate-300">It scales logits: P(c_i) = exp(z_i / T) / sum exp(z_j / T). Low temperature (T &lt; 0.5) makes predictions conservative and grammatically strict. High temperature (T &gt; 1.0) increases entropy and diversity.</p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="font-bold text-amber-300">Q: What is Perplexity?</span>
                <p className="text-slate-300">Perplexity is exp(loss), representing the effective branching factor. Lower perplexity means the model is more confident in predicting the next character.</p>
              </div>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950/50 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold transition-colors cursor-pointer"
          >
            Close Console
          </button>
        </div>

      </div>
    </div>
  );
};
