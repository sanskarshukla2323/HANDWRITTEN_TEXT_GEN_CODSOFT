import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, CheckCircle, Sparkles, TrendingDown, Clock, Cpu, Download } from 'lucide-react';
import { CharRnnEngine } from '../../ml/charRnnEngine';
import { TrainingConfig } from '../../types/rnn';

interface TrainingTabProps {
  engine: CharRnnEngine;
  currentText: string;
  onModelUpdated: () => void;
}

export const TrainingTab: React.FC<TrainingTabProps> = ({
  engine,
  currentText,
  onModelUpdated,
}) => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(engine.totalEpochsTrained);
  const [epochsToTrain, setEpochsToTrain] = useState<number>(25);
  const [hiddenUnits, setHiddenUnits] = useState<number>(engine.config.hiddenUnits);
  const [learningRate, setLearningRate] = useState<number>(engine.config.learningRate);
  const [architecture, setArchitecture] = useState<'SimpleRNN' | 'LSTM'>(engine.config.architecture);
  const [liveSample, setLiveSample] = useState<string>('');
  const [lossData, setLossData] = useState(engine.lossHistory);

  const isTrainingRef = useRef(false);
  isTrainingRef.current = isTraining;

  // Sync parameters if engine config changed
  useEffect(() => {
    setCurrentEpoch(engine.totalEpochsTrained);
    setLossData([...engine.lossHistory]);
  }, [engine]);

  // Total parameter count calculation
  const totalParams = React.useMemo(() => {
    const H = hiddenUnits;
    const V = engine.vocab.vocabSize;
    // W_xh [H * V] + W_hh [H * H] + b_h [H] + W_hy [V * H] + b_y [V]
    return (H * V) + (H * H) + H + (V * H) + V;
  }, [hiddenUnits, engine.vocab.vocabSize]);

  // Train a single epoch asynchronously
  const trainOneEpoch = (epochNum: number) => {
    const seqLen = engine.config.sequenceLength;
    const steps = Math.min(600, currentText.length - seqLen - 1);
    let h = new Array(hiddenUnits).fill(0);
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

    const newRecord = {
      epoch: engine.totalEpochsTrained,
      loss: Number(avgLoss.toFixed(4)),
      perplexity: Number(perp.toFixed(2)),
      elapsedMs: Date.now(),
    };

    engine.lossHistory.push(newRecord);
    setLossData([...engine.lossHistory]);
    setCurrentEpoch(engine.totalEpochsTrained);

    // Sample a snippet of text every 3 epochs
    if (epochNum % 3 === 0 || epochNum === 1) {
      const seed = currentText.slice(0, 15);
      const sample = engine.generate({ seed, length: 70, temperature: 0.6, topK: 5 });
      setLiveSample(sample.text);
    }

    onModelUpdated();
  };

  // Training loop runner
  const startTrainingLoop = () => {
    setIsTraining(true);
    let epoch = 0;

    const step = () => {
      if (!isTrainingRef.current || epoch >= epochsToTrain) {
        setIsTraining(false);
        return;
      }
      epoch++;
      trainOneEpoch(epoch);
      setTimeout(step, 60); // Give browser render time
    };

    setTimeout(step, 10);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
  };

  const handleResetModel = () => {
    setIsTraining(false);
    engine.config.hiddenUnits = hiddenUnits;
    engine.config.learningRate = learningRate;
    engine.config.architecture = architecture;
    engine.initWeights();
    setCurrentEpoch(0);
    setLossData([]);
    setLiveSample('');
    onModelUpdated();
  };

  const handleLoadPretrained = () => {
    setIsTraining(false);
    engine.loadPretrainedWeights(currentText);
    setCurrentEpoch(engine.totalEpochsTrained);
    setLossData([...engine.lossHistory]);
    const seed = currentText.slice(0, 15);
    const sample = engine.generate({ seed, length: 80, temperature: 0.6, topK: 5 });
    setLiveSample(sample.text);
    onModelUpdated();
  };

  // SVG Chart Dimensions & Coordinates
  const chartWidth = 650;
  const chartHeight = 220;
  const padding = { top: 20, right: 30, bottom: 30, left: 45 };

  const maxLoss = lossData.length > 0 ? Math.max(...lossData.map(d => d.loss), 3.0) : 4.0;
  const minLoss = lossData.length > 0 ? Math.min(...lossData.map(d => d.loss), 0.5) : 0.0;

  const points = lossData.map((d, idx) => {
    const x = padding.left + (idx / Math.max(1, lossData.length - 1)) * (chartWidth - padding.left - padding.right);
    const yRange = maxLoss - minLoss || 1;
    const y = chartHeight - padding.bottom - ((d.loss - minLoss) / yRange) * (chartHeight - padding.top - padding.bottom);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">03. Model Architecture & Live Training</h2>
        <p className="text-xs text-slate-400 mt-1">
          Train a genuine Character-Level RNN or LSTM on the active text corpus. The network computes categorical cross-entropy loss and backpropagates gradients through time (BPTT).
        </p>
      </div>

      {/* Hyperparameter Controls */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <span>Hyperparameter Configuration</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div>
            <label className="text-slate-400 block mb-1">Architecture:</label>
            <select
              value={architecture}
              disabled={isTraining}
              onChange={(e) => setArchitecture(e.target.value as 'SimpleRNN' | 'LSTM')}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="SimpleRNN">SimpleRNN (Elman)</option>
              <option value="LSTM">LSTM (Gated Cell)</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Hidden Units ($H$):</label>
            <select
              value={hiddenUnits}
              disabled={isTraining}
              onChange={(e) => setHiddenUnits(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value={32}>32 units (Ultra Fast)</option>
              <option value={64}>64 units (Balanced)</option>
              <option value={128}>128 units (Higher Capacity)</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Epochs to Run:</label>
            <input
              type="number"
              min={5}
              max={100}
              value={epochsToTrain}
              disabled={isTraining}
              onChange={(e) => setEpochsToTrain(Math.max(1, Number(e.target.value)))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 font-mono-code text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="text-slate-400 block mb-1">Learning Rate:</label>
            <select
              value={learningRate}
              disabled={isTraining}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 font-mono-code text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value={0.01}>0.01 (Aggressive)</option>
              <option value={0.005}>0.005 (Recommended)</option>
              <option value={0.001}>0.001 (Conservative)</option>
            </select>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-slate-800">
          {!isTraining ? (
            <button
              onClick={startTrainingLoop}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{currentEpoch > 0 ? 'Continue Training' : 'Start Training'}</span>
            </button>
          ) : (
            <button
              onClick={handleStopTraining}
              className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer"
            >
              <Pause className="w-3.5 h-3.5 fill-current" />
              <span>Pause Training</span>
            </button>
          )}

          <button
            onClick={() => trainOneEpoch(1)}
            disabled={isTraining}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium transition-colors cursor-pointer"
          >
            <FastForward className="w-3.5 h-3.5" />
            <span>Step 1 Epoch</span>
          </button>

          <button
            onClick={handleLoadPretrained}
            disabled={isTraining}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-indigo-300 border border-indigo-900/40 rounded-lg text-xs font-medium transition-colors cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load Quick Pretrained Checkpoint</span>
          </button>

          <button
            onClick={handleResetModel}
            disabled={isTraining}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-400 hover:text-slate-200 border border-slate-700 rounded-lg text-xs font-medium transition-colors ml-auto cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Weights</span>
          </button>
        </div>
      </div>

      {/* Real-Time Training Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Current Loss (Cross-Entropy)</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {engine.currentLoss > 0 ? engine.currentLoss.toFixed(4) : '—'}
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Perplexity ($e^{'{loss}'}$)</span>
          <span className="text-xl font-bold font-mono-code text-indigo-400 tabular-nums">
            {engine.currentPerplexity > 0 ? engine.currentPerplexity.toFixed(2) : '—'}
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Epochs Trained</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {currentEpoch}
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Model Parameters</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {totalParams.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Live Loss Graph & Live Sample Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Cols: Loss Curve Chart */}
        <div className="lg:col-span-2 bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-indigo-400" />
              <span>Training Loss Curve Progression</span>
            </h3>
            <span className="text-[11px] text-slate-400 font-mono-code">
              {lossData.length > 0 ? `${lossData.length} checkpoints recorded` : 'Awaiting training...'}
            </span>
          </div>

          <div className="w-full bg-slate-900/90 border border-slate-800 rounded-lg p-2 overflow-x-auto">
            {lossData.length > 1 ? (
              <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-52">
                {/* Horizontal Grid lines */}
                {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
                  const y = padding.top + pct * (chartHeight - padding.top - padding.bottom);
                  const val = (maxLoss - pct * (maxLoss - minLoss)).toFixed(2);
                  return (
                    <g key={i}>
                      <line
                        x1={padding.left}
                        y1={y}
                        x2={chartWidth - padding.right}
                        y2={y}
                        stroke="rgba(255, 255, 255, 0.08)"
                        strokeDasharray="4 4"
                      />
                      <text x={padding.left - 8} y={y + 3} textAnchor="end" className="text-[9px] fill-slate-500 font-mono">
                        {val}
                      </text>
                    </g>
                  );
                })}

                {/* Loss Line Path */}
                <polyline
                  fill="none"
                  stroke="#818cf8"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={points}
                />

                {/* Point Dots */}
                {lossData.map((d, idx) => {
                  const x = padding.left + (idx / Math.max(1, lossData.length - 1)) * (chartWidth - padding.left - padding.right);
                  const yRange = maxLoss - minLoss || 1;
                  const y = chartHeight - padding.bottom - ((d.loss - minLoss) / yRange) * (chartHeight - padding.top - padding.bottom);
                  return (
                    <circle
                      key={idx}
                      cx={x}
                      cy={y}
                      r="3.5"
                      className="fill-indigo-400 stroke-slate-900 stroke-2 hover:r-5 transition-all"
                    >
                      <title>Epoch {d.epoch}: Loss = {d.loss}</title>
                    </circle>
                  );
                })}
              </svg>
            ) : (
              <div className="h-52 flex flex-col items-center justify-center text-slate-500 text-xs">
                <TrendingDown className="w-8 h-8 mb-2 opacity-40 text-indigo-400" />
                <span>Click "Start Training" or "Load Quick Pretrained Checkpoint" to see the live loss curve.</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Live Training Text Preview */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3 flex flex-col">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span>Live Learning Output</span>
            </h3>
            <span className="text-[10px] text-indigo-300 font-mono-code">Sampled on epoch</span>
          </div>

          <p className="text-xs text-slate-400">
            As the neural network trains, you will witness it transition from random characters to phonetic syllables and recognized English words:
          </p>

          <div className="flex-1 bg-slate-900 border border-slate-700/60 rounded-lg p-3 font-mono-code text-xs text-amber-300/90 leading-relaxed overflow-y-auto max-h-48">
            {liveSample ? (
              <span>"{liveSample}..."</span>
            ) : (
              <span className="text-slate-500 italic">No samples generated yet. Start training to observe character emergence.</span>
            )}
          </div>

          <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Vocabulary: {engine.vocab.vocabSize} symbols</span>
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle className="w-3 h-3" />
              <span>Weights Active</span>
            </span>
          </div>
        </div>

      </div>

    </div>
  );
};
