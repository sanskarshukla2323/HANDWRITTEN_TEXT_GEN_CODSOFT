/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useId } from 'react';
import { 
  Sparkles, 
  Upload, 
  FileText, 
  Download, 
  Copy, 
  Check, 
  RotateCcw, 
  Sliders, 
  Layers, 
  ChevronLeft, 
  ChevronRight, 
  Cpu, 
  FileCheck,
  AlertCircle
} from 'lucide-react';
import { CharRnnEngine, buildCharVocab } from './ml/charRnnEngine';
import { SAMPLE_DATASETS } from './data/sampleDatasets';
import { 
  PageConfig, 
  PAGE_DIMENSIONS, 
  paginateText, 
  drawHandwrittenPage, 
  exportToPdf 
} from './utils/handwritingRenderer';
import { MlDrawerModal } from './components/MlDrawerModal';

export default function App() {
  const [inputText, setInputText] = useState<string>(
    "Handwriting is an intimate mirror of the human spirit. When ink touches paper, thought becomes tangible form. Each curve, stroke, and flourish carries the rhythm of breath and contemplation. Unlike mechanical typography, cursive manuscript reveals the subtle hesitation, speed, and emotion of the writer. Throughout the centuries, scholars kept leather journals filled with ink annotations."
  );

  // Workflow modes
  const [generationMode, setGenerationMode] = useState<'rnn-continuation' | 'direct'>('rnn-continuation');
  const [outputCharsToAdd, setOutputCharsToAdd] = useState<number>(180);
  const [temperature, setTemperature] = useState<number>(0.7);

  // Page & Style controls
  const [pageSize, setPageSize] = useState<'A4' | 'Letter' | 'Note'>('A4');
  const [paperStyle, setPaperStyle] = useState<'lined' | 'plain' | 'grid' | 'sepia'>('lined');
  const [inkColor, setInkColor] = useState<string>('#1e3a8a'); // Royal fountain blue
  const [handwritingFont, setHandwritingFont] = useState<string>('Caveat');
  const [fontSize, setFontSize] = useState<number>(26);
  const [lineHeight, setLineHeight] = useState<number>(34);
  const [humanJitter, setHumanJitter] = useState<number>(1.8);

  // Output & Pagination state
  const [fullOutputText, setFullOutputText] = useState<string>(inputText);
  const [paginatedLines, setPaginatedLines] = useState<string[][]>([['']]);
  const [currentPageIndex, setCurrentPageIndex] = useState<number>(0);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [isExportingPdf, setIsExportingPdf] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputId = useId();

  // Initialize persistent RNN engine
  const engineRef = useRef<CharRnnEngine | null>(null);
  if (!engineRef.current) {
    const vocab = buildCharVocab(SAMPLE_DATASETS[0].text);
    engineRef.current = new CharRnnEngine(vocab, {
      architecture: 'LSTM',
      hiddenUnits: 64,
      sequenceLength: 25,
      learningRate: 0.005,
    });
    engineRef.current.loadPretrainedWeights(SAMPLE_DATASETS[0].text);
  }

  // Predefined Ink Colors
  const inkOptions = [
    { name: 'Royal Blue', value: '#1e3a8a' },
    { name: 'Carbon Black', value: '#0f172a' },
    { name: 'Dark Sepia', value: '#5c2e0b' },
    { name: 'Crimson Red', value: '#991b1b' },
  ];

  const fontOptions = [
    { name: 'Fountain Pen Cursive (Caveat)', value: 'Caveat' },
    { name: 'Vintage Manuscript (Homemade Apple)', value: 'Homemade Apple' },
    { name: 'Casual Ballpoint (Cedarville)', value: 'Cedarville Cursive' },
    { name: 'Architect Script (Shadows Into Light)', value: 'Shadows Into Light' },
  ];

  // Helper to paginate and draw canvas
  const updateCanvasRendering = (textToRender: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = PAGE_DIMENSIONS[pageSize];
    canvas.width = width;
    canvas.height = height;

    ctx.font = `${fontSize}px "${handwritingFont}", cursive`;
    const pages = paginateText(textToRender, ctx, width, height, fontSize, lineHeight, paperStyle);
    setPaginatedLines(pages);

    const safePageIndex = Math.min(currentPageIndex, pages.length - 1);
    setCurrentPageIndex(safePageIndex);

    const config: PageConfig = {
      pageSize,
      paperStyle,
      inkColor,
      handwritingFont,
      fontSize,
      lineHeight,
      humanJitter,
    };

    drawHandwrittenPage(ctx, pages[safePageIndex] || [], safePageIndex, pages.length, config);
  };

  // Initial draw and redraw when styles change
  useEffect(() => {
    updateCanvasRendering(fullOutputText);
  }, [pageSize, paperStyle, inkColor, handwritingFont, fontSize, lineHeight, humanJitter, currentPageIndex]);

  // Main Action: Generate Handwriting
  const handleGenerateHandwriting = () => {
    setIsGenerating(true);

    if (generationMode === 'rnn-continuation' && engineRef.current) {
      // Use Character-Level RNN to auto-complete and generate sequence from seed
      const seed = inputText.trim().length > 0 ? inputText : 'Handwriting is ';
      
      // Update vocab if new characters were typed
      const currentVocab = buildCharVocab(seed + SAMPLE_DATASETS[0].text);
      if (engineRef.current.vocab.vocabSize !== currentVocab.vocabSize) {
        engineRef.current.vocab = currentVocab;
        engineRef.current.loadPretrainedWeights(seed + SAMPLE_DATASETS[0].text);
      }

      setTimeout(() => {
        const result = engineRef.current!.generate({
          seed,
          length: outputCharsToAdd,
          temperature,
          topK: 8,
        });

        setFullOutputText(result.text);
        setCurrentPageIndex(0);
        updateCanvasRendering(result.text);
        setIsGenerating(false);
      }, 50);
    } else {
      // Direct rendering of exact user input
      setFullOutputText(inputText);
      setCurrentPageIndex(0);
      updateCanvasRendering(inputText);
      setIsGenerating(false);
    }
  };

  // Handle .txt file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content && content.trim().length > 0) {
        setInputText(content);
        setFullOutputText(content);
        updateCanvasRendering(content);
      }
    };
    reader.readAsText(file);
  };

  // Export Canvas to PNG
  const handleDownloadPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `handwritten_page_${currentPageIndex + 1}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Export All Pages to PDF
  const handleDownloadPdf = async () => {
    try {
      setIsExportingPdf(true);
      const config: PageConfig = {
        pageSize,
        paperStyle,
        inkColor,
        handwritingFont,
        fontSize,
        lineHeight,
        humanJitter,
      };
      await exportToPdf(paginatedLines, config, 'handwritten_document.pdf');
    } finally {
      setIsExportingPdf(false);
    }
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(fullOutputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
      
      {/* 3-Zone Header Bar */}
      <header className="sticky top-0 z-40 bg-slate-900/95 backdrop-blur border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          
          {/* Zone 1: Brand Wordmark */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-sm font-bold text-lg">
              ✍️
            </div>
            <div>
              <span className="text-base font-bold tracking-tight text-white whitespace-nowrap block">
                Handwriting Studio
              </span>
              <span className="text-[11px] text-slate-400 whitespace-nowrap block">
                Text-to-Handwriting Generator · Character-Level RNN
              </span>
            </div>
          </div>

          {/* Zone 2 / Primary Action: ML Engine Info */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setIsModalOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white text-xs font-semibold shadow-sm transition-colors cursor-pointer"
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>ML Engine & Kaggle Info</span>
            </button>
          </div>

        </div>
      </header>

      {/* Main Single-Page Workflow */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {/* Step Banner */}
        <div className="mb-6 p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-3">
            <span className="font-bold text-indigo-400 font-mono-code uppercase">Task 2 Workflow:</span>
            <div className="flex items-center gap-2 text-slate-300">
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-medium">1. Enter / Upload Text</span>
              <span className="text-slate-500">→</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-medium">2. Generate Handwriting</span>
              <span className="text-slate-500">→</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-medium">3. Preview & Download</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-400">Architecture:</span>
            <span className="text-indigo-300 font-medium">Stage A (RNN Text Modeling) ➔ Stage B (Handwriting Synthesis)</span>
          </div>
        </div>

        {/* 2-Column Responsive Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT COLUMN: Input & Generation Controls (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* 1. Input Text Area */}
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-indigo-400" />
                  <span>1. Input Text or Seed Prompt</span>
                </label>
                
                <div className="flex items-center gap-2">
                  <label
                    htmlFor={fileInputId}
                    className="flex items-center gap-1 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-[11px] font-medium cursor-pointer transition-colors"
                  >
                    <Upload className="w-3 h-3" />
                    <span>Upload .txt</span>
                  </label>
                  <input
                    id={fileInputId}
                    type="file"
                    accept=".txt"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>
              </div>

              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                rows={5}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-100 font-mono-code focus:outline-none focus:border-indigo-500 leading-relaxed resize-y"
                placeholder="Type or paste the text you want to transform into handwriting..."
              />

              <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
                <span>{inputText.length} characters</span>
                <div className="flex items-center gap-1.5">
                  <span>Samples:</span>
                  <button
                    onClick={() => setInputText(SAMPLE_DATASETS[0].text)}
                    className="hover:text-indigo-300 underline cursor-pointer"
                  >
                    Essay
                  </button>
                  <span>·</span>
                  <button
                    onClick={() => setInputText(SAMPLE_DATASETS[1].text)}
                    className="hover:text-indigo-300 underline cursor-pointer"
                  >
                    Alice
                  </button>
                  <span>·</span>
                  <button
                    onClick={() => setInputText(SAMPLE_DATASETS[2].text)}
                    className="hover:text-indigo-300 underline cursor-pointer"
                  >
                    ML Notes
                  </button>
                </div>
              </div>
            </div>

            {/* 2. Generation Mode & RNN Settings */}
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-400" />
                  <span>2. Stage A: Text Processing Mode</span>
                </h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 font-mono-code font-bold">
                  LSTM Active
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <label className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${generationMode === 'rnn-continuation' ? 'bg-indigo-950/30 border-indigo-500/60 text-white' : 'bg-slate-900/60 border-slate-800 text-slate-400'}`}>
                  <input
                    type="radio"
                    name="genMode"
                    checked={generationMode === 'rnn-continuation'}
                    onChange={() => setGenerationMode('rnn-continuation')}
                    className="mt-0.5 accent-indigo-500 cursor-pointer"
                  />
                  <div>
                    <span className="font-semibold block text-slate-200">RNN Auto-Complete</span>
                    <span className="text-[10px] text-slate-400">Uses prompt as seed, generates new characters with RNN</span>
                  </div>
                </label>

                <label className={`flex items-start gap-2.5 p-3 rounded-lg border cursor-pointer transition-colors ${generationMode === 'direct' ? 'bg-indigo-950/30 border-indigo-500/60 text-white' : 'bg-slate-900/60 border-slate-800 text-slate-400'}`}>
                  <input
                    type="radio"
                    name="genMode"
                    checked={generationMode === 'direct'}
                    onChange={() => setGenerationMode('direct')}
                    className="mt-0.5 accent-indigo-500 cursor-pointer"
                  />
                  <div>
                    <span className="font-semibold block text-slate-200">Direct Rendering</span>
                    <span className="text-[10px] text-slate-400">Renders the exact input text into handwriting</span>
                  </div>
                </label>
              </div>

              {generationMode === 'rnn-continuation' && (
                <div className="space-y-3 pt-2 border-t border-slate-800 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300 font-medium">Output Length (Tokens to Generate):</span>
                    <span className="text-indigo-400 font-mono-code font-bold tabular-nums">+{outputCharsToAdd} chars</span>
                  </div>
                  <input
                    type="range"
                    min={50}
                    max={400}
                    step={25}
                    value={outputCharsToAdd}
                    onChange={(e) => setOutputCharsToAdd(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />

                  <div className="flex items-center justify-between pt-1">
                    <span className="text-slate-300 font-medium">Sampling Temperature ($T$):</span>
                    <span className="text-indigo-400 font-mono-code font-bold tabular-nums">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.2}
                    max={1.4}
                    step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>0.2 (Conservative)</span>
                    <span>0.7 (Natural)</span>
                    <span>1.4 (Creative)</span>
                  </div>
                </div>
              )}
            </div>

            {/* 3. Stage B: Handwriting Document Style Controls */}
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
              <h3 className="text-xs font-bold text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span>3. Stage B: Document & Handwriting Styling</span>
              </h3>

              <div className="grid grid-cols-2 gap-3 text-xs">
                {/* Page Size */}
                <div>
                  <label className="text-slate-400 block mb-1">Page Format:</label>
                  <select
                    value={pageSize}
                    onChange={(e) => setPageSize(e.target.value as any)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                  >
                    <option value="A4">A4 (210 × 297 mm)</option>
                    <option value="Letter">US Letter (8.5 × 11 in)</option>
                    <option value="Note">Compact Notebook</option>
                  </select>
                </div>

                {/* Paper Style */}
                <div>
                  <label className="text-slate-400 block mb-1">Paper Surface:</label>
                  <select
                    value={paperStyle}
                    onChange={(e) => setPaperStyle(e.target.value as any)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                  >
                    <option value="lined">Lined Journal (Margin Rule)</option>
                    <option value="plain">Plain Clean Parchment</option>
                    <option value="grid">Engineering Graph Paper</option>
                    <option value="sepia">Vintage Aged Sepia</option>
                  </select>
                </div>

                {/* Handwriting Stroke Style */}
                <div className="col-span-2">
                  <label className="text-slate-400 block mb-1">Handwriting Stroke Font:</label>
                  <select
                    value={handwritingFont}
                    onChange={(e) => setHandwritingFont(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
                  >
                    {fontOptions.map(f => (
                      <option key={f.value} value={f.value}>{f.name}</option>
                    ))}
                  </select>
                </div>

                {/* Ink Color Selector */}
                <div className="col-span-2">
                  <label className="text-slate-400 block mb-1.5">Fountain Pen Ink Color:</label>
                  <div className="flex items-center gap-3">
                    {inkOptions.map(ink => (
                      <button
                        key={ink.value}
                        onClick={() => setInkColor(ink.value)}
                        className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border text-xs cursor-pointer transition-all ${inkColor === ink.value ? 'bg-slate-800 border-indigo-400 text-white shadow-sm' : 'border-slate-800 text-slate-400 hover:bg-slate-800/40'}`}
                      >
                        <span className="w-3.5 h-3.5 rounded-full shrink-0 border border-white/20" style={{ backgroundColor: ink.value }} />
                        <span>{ink.name}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Font Size & Jitter */}
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Pen Size:</span>
                    <span className="font-mono-code text-indigo-400">{fontSize}px</span>
                  </div>
                  <input
                    type="range"
                    min={20}
                    max={36}
                    value={fontSize}
                    onChange={(e) => {
                      setFontSize(Number(e.target.value));
                      setLineHeight(Math.round(Number(e.target.value) * 1.35));
                    }}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Human Jitter:</span>
                    <span className="font-mono-code text-indigo-400">{humanJitter.toFixed(1)}px</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={4}
                    step={0.5}
                    value={humanJitter}
                    onChange={(e) => setHumanJitter(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* PRIMARY WORKFLOW BUTTON: GENERATE HANDWRITING */}
            <div>
              <button
                onClick={handleGenerateHandwriting}
                disabled={isGenerating}
                className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 active:from-indigo-700 active:to-violet-700 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20 transition-all cursor-pointer disabled:opacity-60"
              >
                <Sparkles className="w-4 h-4 fill-current" />
                <span>{isGenerating ? 'Synthesizing Handwriting...' : 'Generate Handwriting'}</span>
              </button>
            </div>

          </div>

          {/* RIGHT COLUMN: Handwritten Page Preview & Downloads (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            
            {/* Action Bar & Downloads */}
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="text-xs font-bold text-white block">Handwritten Page Preview</span>
                <span className="text-[11px] text-slate-400">
                  {PAGE_DIMENSIONS[pageSize].name} · {paginatedLines.length} {paginatedLines.length === 1 ? 'Page' : 'Pages'}
                </span>
              </div>

              {/* Download Buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownloadPng}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 rounded-lg text-xs font-semibold transition-colors cursor-pointer shadow-sm"
                  title="Download current page as high-resolution PNG image"
                >
                  <Download className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Download PNG</span>
                </button>

                <button
                  onClick={handleDownloadPdf}
                  disabled={isExportingPdf}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold transition-colors cursor-pointer shadow-sm"
                  title="Download all pages combined in a multi-page PDF"
                >
                  <FileCheck className="w-3.5 h-3.5" />
                  <span>{isExportingPdf ? 'Exporting PDF...' : 'Download PDF'}</span>
                </button>

                <button
                  onClick={handleCopyText}
                  className="p-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded-lg text-xs transition-colors cursor-pointer"
                  title="Copy text content"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Canvas Preview Container */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 sm:p-6 flex flex-col items-center justify-center overflow-hidden shadow-2xl relative">
              
              {/* Pagination Controls */}
              {paginatedLines.length > 1 && (
                <div className="w-full flex items-center justify-between mb-3 px-2 text-xs text-slate-400 font-mono-code">
                  <button
                    onClick={() => setCurrentPageIndex(prev => Math.max(0, prev - 1))}
                    disabled={currentPageIndex === 0}
                    className="flex items-center gap-1 px-2.5 py-1 bg-slate-850 hover:bg-slate-800 disabled:opacity-30 rounded border border-slate-800 cursor-pointer"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Previous</span>
                  </button>

                  <span className="font-semibold text-slate-300">
                    Page {currentPageIndex + 1} of {paginatedLines.length}
                  </span>

                  <button
                    onClick={() => setCurrentPageIndex(prev => Math.min(paginatedLines.length - 1, prev + 1))}
                    disabled={currentPageIndex === paginatedLines.length - 1}
                    className="flex items-center gap-1 px-2.5 py-1 bg-slate-850 hover:bg-slate-800 disabled:opacity-30 rounded border border-slate-800 cursor-pointer"
                  >
                    <span>Next</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {/* The HTML5 Canvas displaying the realistic handwritten page */}
              <div className="w-full max-w-[560px] shadow-2xl rounded-sm overflow-hidden border border-slate-700/60 bg-white">
                <canvas
                  ref={canvasRef}
                  className="w-full h-auto block select-none pointer-events-none"
                  style={{ aspectRatio: `${PAGE_DIMENSIONS[pageSize].width} / ${PAGE_DIMENSIONS[pageSize].height}` }}
                />
              </div>

              {/* Caption */}
              <div className="mt-3 text-center text-[11px] text-slate-500">
                High-resolution canvas preview ({PAGE_DIMENSIONS[pageSize].width} × {PAGE_DIMENSIONS[pageSize].height} px). Click "Download PNG" or "Download PDF" to export.
              </div>
            </div>

            {/* Stage A/B Architecture Explanation Footnote */}
            <div className="p-3.5 rounded-xl bg-slate-800/30 border border-slate-800 text-[11px] text-slate-400 space-y-1">
              <span className="font-semibold text-slate-300 block">CODSOFT AI/ML Internship Compliance:</span>
              <p>
                <strong>Stage A:</strong> The genuine Character-Level RNN (LSTM) models character sequence probabilities P(c_t | c_1 ... c_(t-1)).
                <br />
                <strong>Stage B:</strong> The Handwriting Synthesizer maps text tokens to high-resolution ink strokes and paper textures, producing ready-to-print handwritten documents.
              </p>
            </div>

          </div>

        </div>

      </main>

      {/* ML Engine & Kaggle Modal Drawer */}
      <MlDrawerModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        engine={engineRef.current!}
        currentText={inputText}
        onModelUpdated={() => updateCanvasRendering(fullOutputText)}
      />

      {/* Simple Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-5 mt-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div>
            <span className="font-semibold text-slate-400">Handwriting Studio</span> · CODSOFT AI/ML Internship Task 2
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsModalOpen(true)}
              className="hover:text-slate-300 transition-colors cursor-pointer"
            >
              ML Training & Loss Curve
            </button>
            <button
              onClick={() => setIsModalOpen(true)}
              className="hover:text-slate-300 transition-colors cursor-pointer"
            >
              Kaggle english.csv Analysis
            </button>
          </div>
        </div>
      </footer>

    </div>
  );
}
