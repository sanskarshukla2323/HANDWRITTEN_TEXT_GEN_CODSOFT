import React, { useState, useId } from 'react';
import { Upload, FileText, Check, AlertCircle, Sparkles, Sliders, Table, RefreshCw } from 'lucide-react';
import { SAMPLE_DATASETS, SampleCorpus } from '../../data/sampleDatasets';
import { analyzeDatasetText, buildCharVocab } from '../../ml/charRnnEngine';
import { CharVocab, DatasetStats } from '../../types/rnn';

interface DatasetTabProps {
  currentText: string;
  onTextChange: (newText: string) => void;
  vocab: CharVocab;
  stats: DatasetStats;
  sequenceLength: number;
  onSequenceLengthChange: (length: number) => void;
}

export const DatasetTab: React.FC<DatasetTabProps> = ({
  currentText,
  onTextChange,
  vocab,
  stats,
  sequenceLength,
  onSequenceLengthChange,
}) => {
  const [selectedCorpusId, setSelectedCorpusId] = useState<string>('handwriting-essay');
  const [lowercase, setLowercase] = useState<boolean>(false);
  const [removeNonAscii, setRemoveNonAscii] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [stepSize, setStepSize] = useState<number>(3);
  const fileInputId = useId();

  const handleSelectPreset = (corpus: SampleCorpus) => {
    setSelectedCorpusId(corpus.id);
    setUploadError(null);
    let txt = corpus.text;
    if (lowercase) txt = txt.toLowerCase();
    onTextChange(txt);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    const reader = new FileReader();

    if (file.name.endsWith('.csv')) {
      reader.onload = (event) => {
        try {
          const content = event.target?.result as string;
          const lines = content.split('\n');
          if (lines.length < 2) {
            setUploadError('CSV file has no data rows.');
            return;
          }
          // Simple CSV line parser: grab largest column content
          const rows = lines.map(line => line.split(','));
          const colCount = rows[0].length;
          let bestColIdx = 0;
          let maxLen = 0;

          for (let c = 0; c < colCount; c++) {
            const colLen = rows.reduce((acc, row) => acc + (row[c]?.length || 0), 0);
            if (colLen > maxLen) {
              maxLen = colLen;
              bestColIdx = c;
            }
          }

          const extracted = rows.slice(1).map(r => r[bestColIdx] || '').join(' ').trim();
          if (extracted.length < 50) {
            setUploadError('Extracted CSV column text is too short (less than 50 characters).');
            return;
          }

          setSelectedCorpusId('custom-csv');
          onTextChange(extracted);
        } catch (err) {
          setUploadError(`Failed to parse CSV: ${String(err)}`);
        }
      };
      reader.readAsText(file);
    } else {
      // Plain text file (.txt)
      reader.onload = (event) => {
        const content = event.target?.result as string;
        if (!content || content.trim().length === 0) {
          setUploadError('Uploaded text file is empty.');
          return;
        }
        if (content.trim().length < 50) {
          setUploadError('Uploaded text file is too short (minimum 50 characters required for sequence training).');
          return;
        }
        setSelectedCorpusId('custom-txt');
        onTextChange(content);
      };
      reader.readAsText(file);
    }
  };

  // Generate sequence window previews
  const sequencePairs = React.useMemo(() => {
    const pairs: { input: string; target: string }[] = [];
    const maxPreviews = 8;
    for (let i = 0; i < currentText.length - sequenceLength && pairs.length < maxPreviews; i += stepSize) {
      pairs.push({
        input: currentText.slice(i, i + sequenceLength),
        target: currentText[i + sequenceLength],
      });
    }
    return pairs;
  }, [currentText, sequenceLength, stepSize]);

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">02. Dataset Ingestion & Preprocessing</h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload a custom training text file or select from verified reference corpora. Inspect the generated character vocabulary and $(X \to y)$ sliding windows.
        </p>
      </div>

      {/* Preset Corpora Selector */}
      <div className="space-y-3">
        <label className="text-xs font-semibold text-slate-300 block">Choose a Built-In Reference Corpus:</label>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {SAMPLE_DATASETS.map((corpus) => {
            const isSelected = selectedCorpusId === corpus.id;
            return (
              <button
                key={corpus.id}
                onClick={() => handleSelectPreset(corpus)}
                className={`text-left p-4 rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-slate-800 border-indigo-500 shadow-sm'
                    : 'bg-slate-800/40 border-slate-700/60 hover:bg-slate-800/80 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span className="text-[11px] font-semibold text-indigo-400 uppercase tracking-wider">
                    {corpus.category}
                  </span>
                  {isSelected && <Check className="w-4 h-4 text-indigo-400 shrink-0" />}
                </div>
                <h4 className="text-xs font-bold text-white line-clamp-1">{corpus.title}</h4>
                <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 leading-normal">
                  {corpus.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* File Upload Box */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Upload className="w-4 h-4 text-indigo-400" />
              <span>Upload Custom Dataset (.txt or .csv)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Supports plain UTF-8 text documents or CSV tables with text columns.
            </p>
          </div>

          <div>
            <label
              htmlFor={fileInputId}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg cursor-pointer inline-flex items-center gap-2 transition-colors shadow-sm"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Browse File</span>
            </label>
            <input
              id={fileInputId}
              type="file"
              accept=".txt,.csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>
        </div>

        {uploadError && (
          <div className="mt-3 p-3 rounded-lg bg-red-950/40 border border-red-800/60 flex items-center gap-2 text-red-300 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>

      {/* Dataset Statistics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Total Characters</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {stats.totalChars.toLocaleString()}
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Unique Vocabulary ($V$)</span>
          <span className="text-xl font-bold font-mono-code text-indigo-400 tabular-nums">
            {stats.uniqueChars} symbols
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Word Count</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {stats.wordCount.toLocaleString()}
          </span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
          <span className="text-[11px] text-slate-400 block mb-1">Line Count</span>
          <span className="text-xl font-bold font-mono-code text-white tabular-nums">
            {stats.lineCount.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Text Area & Vocabulary Inspector Split */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Left Column: Cleaned Text Editor / Preview */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Active Text Corpus</h3>
            <span className="text-[11px] text-slate-400">{currentText.length} characters</span>
          </div>

          <textarea
            value={currentText}
            onChange={(e) => onTextChange(e.target.value)}
            rows={10}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs font-mono-code text-slate-200 focus:outline-none focus:border-indigo-500 leading-relaxed resize-y"
            placeholder="Type or paste custom text here to train the character-level RNN..."
          />

          {/* Quick Sanitization Helpers */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <button
              onClick={() => onTextChange(currentText.toLowerCase())}
              className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded text-[11px] font-medium transition-colors cursor-pointer"
            >
              Convert to Lowercase
            </button>
            <button
              onClick={() => onTextChange(currentText.replace(/[^\x00-\x7F]/g, ''))}
              className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded text-[11px] font-medium transition-colors cursor-pointer"
            >
              Strip Non-ASCII Symbols
            </button>
            <button
              onClick={() => onTextChange(currentText.replace(/\s+/g, ' ').trim())}
              className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded text-[11px] font-medium transition-colors cursor-pointer"
            >
              Normalize Whitespace
            </button>
          </div>
        </div>

        {/* Right Column: Character Vocabulary Table */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Table className="w-4 h-4 text-indigo-400" />
              <span>Character Vocabulary Mapping ({vocab.vocabSize} tokens)</span>
            </h3>
            <span className="text-[11px] text-slate-400">Tokens indexed 0 to {vocab.vocabSize - 1}</span>
          </div>

          <div className="max-h-72 overflow-y-auto border border-slate-700/60 rounded-lg bg-slate-900/60 text-xs">
            <table className="w-full text-left border-collapse">
              <thead className="bg-slate-800/90 sticky top-0 border-b border-slate-700 text-[11px] text-slate-400 font-semibold">
                <tr>
                  <th className="py-2 px-3">Index</th>
                  <th className="py-2 px-3">Character</th>
                  <th className="py-2 px-3">ASCII/Unicode</th>
                  <th className="py-2 px-3">Frequency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono-code">
                {vocab.chars.map((char, idx) => {
                  const count = currentText.split(char).length - 1;
                  const displayChar = char === ' ' ? '<SPACE>' : char === '\n' ? '<NEWLINE>' : char === '\t' ? '<TAB>' : char;
                  return (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-1.5 px-3 text-indigo-400 font-semibold">{idx}</td>
                      <td className="py-1.5 px-3 text-white font-bold bg-slate-800/40 rounded px-1">{displayChar}</td>
                      <td className="py-1.5 px-3 text-slate-400">{char.charCodeAt(0)}</td>
                      <td className="py-1.5 px-3 text-slate-300 tabular-nums">{count.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {/* Sliding Window Sequence ($X \to y$) Visualizer */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span>Input-Output Sequence Tensor Inspector ($X \to y$)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              The model takes $L$ past characters as input $X$ and trains to predict the single next character $y$.
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Sequence Length ($L$):</span>
              <span className="text-indigo-400 font-bold font-mono-code">{sequenceLength}</span>
              <input
                type="range"
                min={10}
                max={50}
                value={sequenceLength}
                onChange={(e) => onSequenceLengthChange(Number(e.target.value))}
                className="w-24 accent-indigo-500 cursor-pointer"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Step Size:</span>
              <span className="text-indigo-400 font-bold font-mono-code">{stepSize}</span>
              <input
                type="range"
                min={1}
                max={5}
                value={stepSize}
                onChange={(e) => setStepSize(Number(e.target.value))}
                className="w-20 accent-indigo-500 cursor-pointer"
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {sequencePairs.map((pair, idx) => (
            <div key={idx} className="bg-slate-900/80 border border-slate-700/60 rounded-lg p-3 text-xs space-y-2">
              <div className="text-[10px] text-slate-400 font-semibold flex items-center justify-between">
                <span>Sample #{idx + 1}</span>
                <span className="text-indigo-400">Length: {sequenceLength}</span>
              </div>
              <div className="p-2 bg-slate-950/60 rounded border border-slate-800 font-mono-code text-[11px] text-slate-200 break-all">
                "{pair.input}"
              </div>
              <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-800">
                <span className="text-slate-400">Next Target ($y$):</span>
                <span className="font-bold text-amber-400 font-mono-code px-1.5 py-0.5 bg-amber-500/10 rounded">
                  '{pair.target === ' ' ? '<SPACE>' : pair.target}'
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
