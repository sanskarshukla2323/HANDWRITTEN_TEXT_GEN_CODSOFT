import React, { useState } from 'react';
import { Play, Copy, Check, Download, PenTool, Sparkles, Sliders, BarChart2, AlertCircle } from 'lucide-react';
import { CharRnnEngine } from '../../ml/charRnnEngine';
import { NextCharProbability } from '../../types/rnn';
import { TabType } from '../Header';

interface GenerationTabProps {
  engine: CharRnnEngine;
  onNavigateToHandwriting: (text: string) => void;
}

export const GenerationTab: React.FC<GenerationTabProps> = ({
  engine,
  onNavigateToHandwriting,
}) => {
  const [seed, setSeed] = useState<string>('Handwriting is ');
  const [genLength, setGenLength] = useState<number>(200);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topK, setTopK] = useState<number>(8);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedText, setGeneratedText] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);
  const [latestProbs, setLatestProbs] = useState<NextCharProbability[]>([]);

  const handleGenerate = () => {
    setIsGenerating(true);
    setGeneratedText(seed);

    // Call engine generation
    setTimeout(() => {
      const result = engine.generate(
        {
          seed,
          length: genLength,
          temperature,
          topK,
        },
        (currentOutput, nextProbs) => {
          setGeneratedText(currentOutput);
          setLatestProbs(nextProbs);
        }
      );

      setGeneratedText(result.text);
      setLatestProbs(result.topPredictions);
      setIsGenerating(false);
    }, 20);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadTxt = () => {
    const blob = new Blob([generatedText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated_handwriting_text.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const quickSeeds = [
    'Handwriting is ',
    'When ink touches ',
    'Alice was beginning ',
    'Recurrent neural ',
    'Shall I compare ',
  ];

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">04. Interactive Text Generation</h2>
        <p className="text-xs text-slate-400 mt-1">
          Generate new text character-by-character from a starting seed phrase. Inspect next-character probabilities and experiment with temperature sampling.
        </p>
      </div>

      {!engine.isTrained && (
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1 text-xs text-amber-200">
            <span className="font-semibold block">Notice: Model has not completed training yet</span>
            <span>
              Generating with random untrained weights will produce random character noise. For best results, run training in the "Training" tab or click "Load Quick Pretrained Checkpoint" to initialize learned sequence weights.
            </span>
          </div>
        </div>
      )}

      {/* Controls & Configuration */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Sliders className="w-4 h-4 text-indigo-400" />
          <span>Generation Parameters</span>
        </h3>

        {/* Seed Input & Quick Seeds */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-300 block">Starting Seed Phrase:</label>
          <input
            type="text"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs font-mono-code text-white focus:outline-none focus:border-indigo-500"
            placeholder="Type a starting prompt..."
          />
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-[11px] text-slate-400 mr-1">Quick Seeds:</span>
            {quickSeeds.map((qs, i) => (
              <button
                key={i}
                onClick={() => setSeed(qs)}
                className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded text-[11px] transition-colors cursor-pointer"
              >
                "{qs}"
              </button>
            ))}
          </div>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-2">
          {/* Temperature Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold">Temperature ($T$):</span>
              <span className="text-indigo-400 font-bold font-mono-code tabular-nums">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-medium">
              <span>0.1 (Strict)</span>
              <span>0.7 (Natural)</span>
              <span>1.5 (Creative)</span>
            </div>
          </div>

          {/* Length Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold">Length (Characters):</span>
              <span className="text-indigo-400 font-bold font-mono-code tabular-nums">{genLength} chars</span>
            </div>
            <input
              type="range"
              min={50}
              max={500}
              step={25}
              value={genLength}
              onChange={(e) => setGenLength(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-medium">
              <span>50 chars</span>
              <span>250 chars</span>
              <span>500 chars</span>
            </div>
          </div>

          {/* Top-K Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold">Top-K Candidate Pool:</span>
              <span className="text-indigo-400 font-bold font-mono-code tabular-nums">{topK} symbols</span>
            </div>
            <input
              type="range"
              min={1}
              max={Math.min(20, engine.vocab.vocabSize || 10)}
              step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-medium">
              <span>1 (Greedy Argmax)</span>
              <span>8 (Balanced)</span>
              <span>20 (Diverse)</span>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <div className="pt-2">
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isGenerating ? 'Sampling Characters...' : 'Generate Text Sequence'}</span>
          </button>
        </div>
      </div>

      {/* Generation Output & Probability Inspector Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Output Area (2 cols) */}
        <div className="lg:col-span-2 bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3 flex flex-col">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span>Generated Text Result</span>
            </h3>
            <span className="text-[11px] text-slate-400 font-mono-code tabular-nums">
              {generatedText.length} characters
            </span>
          </div>

          <div className="flex-1 min-h-[180px] bg-slate-900 border border-slate-700 rounded-lg p-4 font-mono-code text-xs text-slate-100 leading-relaxed whitespace-pre-wrap select-all">
            {generatedText ? (
              <span>{generatedText}</span>
            ) : (
              <span className="text-slate-500 italic">
                Generated text will appear here. Configure the parameters above and click "Generate Text Sequence".
              </span>
            )}
          </div>

          {/* Action Tools */}
          {generatedText && (
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 rounded text-xs font-medium transition-colors cursor-pointer"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied to Clipboard' : 'Copy Text'}</span>
              </button>

              <button
                onClick={handleDownloadTxt}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 rounded text-xs font-medium transition-colors cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download (.txt)</span>
              </button>

              <button
                onClick={() => onNavigateToHandwriting(generatedText)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-300 rounded text-xs font-semibold transition-colors ml-auto cursor-pointer"
              >
                <PenTool className="w-3.5 h-3.5" />
                <span>Render in Handwriting Canvas</span>
              </button>
            </div>
          )}
        </div>

        {/* Right Col: Next-Character Probability Inspector */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-indigo-400" />
              <span>Next Character Probabilities</span>
            </h3>
          </div>

          <p className="text-[11px] text-slate-400">
            Softmax distribution $P(c_i)$ computed for the final sequence step under temperature $T={temperature.toFixed(2)}$:
          </p>

          <div className="space-y-2 pt-1 font-mono-code text-xs">
            {latestProbs.length > 0 ? (
              latestProbs.map((p, idx) => {
                const pct = (p.prob * 100).toFixed(1);
                const displayChar = p.char === ' ' ? '<SPACE>' : p.char === '\n' ? '<NEWLINE>' : p.char;
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-white font-bold bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">
                        '{displayChar}'
                      </span>
                      <span className="text-indigo-300 font-semibold">{pct}%</span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(2, p.prob * 100))}%` }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="p-4 text-center text-slate-500 text-xs italic">
                Generate text to view the real-time next-character distribution.
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
};
