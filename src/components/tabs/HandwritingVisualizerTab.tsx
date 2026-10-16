import React, { useState, useRef } from 'react';
import { PenTool, Download, Sliders, Palette, RefreshCw, FileText } from 'lucide-react';

interface HandwritingVisualizerTabProps {
  initialText: string;
}

export const HandwritingVisualizerTab: React.FC<HandwritingVisualizerTabProps> = ({ initialText }) => {
  const [text, setText] = useState<string>(
    initialText ||
    "Handwriting is an intimate mirror of the human spirit. When ink touches paper, thought becomes tangible form. Each stroke carries the rhythm of breath and contemplation."
  );

  const [handwritingStyle, setHandwritingStyle] = useState<string>('font-handwriting-caveat');
  const [paperStyle, setPaperStyle] = useState<'lined' | 'grid' | 'parchment' | 'chalkboard'>('lined');
  const [inkColor, setInkColor] = useState<string>('#1e3a8a'); // Royal blue fountain ink
  const [fontSize, setFontSize] = useState<number>(24);
  const [lineHeight, setLineHeight] = useState<number>(28);
  const [letterSpacing, setLetterSpacing] = useState<number>(1);
  const [slantAngle, setSlantAngle] = useState<number>(-2); // Subtle left-to-right slant

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const paperClasses = {
    lined: 'lined-paper text-slate-800 shadow-md',
    grid: 'grid-paper text-slate-800 shadow-md',
    parchment: 'bg-[#faf6ee] text-slate-800 shadow-md border border-[#e5dec9]',
    chalkboard: 'bg-slate-900 text-slate-100 shadow-md border border-slate-700',
  };

  const inkPalette = [
    { name: 'Royal Blue Ink', value: '#1e3a8a' },
    { name: 'Carbon Black Ink', value: '#0f172a' },
    { name: 'Sepia Walnut Ink', value: '#78350f' },
    { name: 'Forest Green Ink', value: '#14532d' },
    { name: 'Crimson Burgundy', value: '#991b1b' },
  ];

  const handleExportPng = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 700;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw paper background
    if (paperStyle === 'chalkboard') {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#faf7ee';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (paperStyle === 'lined') {
        // Red margin line
        ctx.strokeStyle = '#fca5a5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(120, 0);
        ctx.lineTo(120, canvas.height);
        ctx.stroke();

        // Blue horizontal lines
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1;
        for (let y = 60; y < canvas.height; y += 34) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }
      }
    }

    // Draw text with selected handwriting font
    let fontName = 'Caveat';
    if (handwritingStyle.includes('apple')) fontName = 'Homemade Apple';
    else if (handwritingStyle.includes('cedar')) fontName = 'Cedarville Cursive';
    else if (handwritingStyle.includes('shadows')) fontName = 'Shadows Into Light';
    else if (handwritingStyle.includes('reenie')) fontName = 'Reenie Beanie';

    ctx.font = `${fontSize * 1.5}px "${fontName}", cursive`;
    ctx.fillStyle = inkColor;

    const words = text.split(' ');
    let line = '';
    let x = paperStyle === 'lined' ? 140 : 60;
    let y = 80;
    const maxWidth = canvas.width - x - 60;
    const canvasLineHeight = lineHeight * 1.5;

    for (let n = 0; n < words.length; n++) {
      const testLine = line + words[n] + ' ';
      const metrics = ctx.measureText(testLine);
      if (metrics.width > maxWidth && n > 0) {
        ctx.fillText(line, x, y);
        line = words[n] + ' ';
        y += canvasLineHeight;
      } else {
        line = testLine;
      }
    }
    ctx.fillText(line, x, y);

    // Trigger download
    const dataUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = 'handwritten_document.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">05. Handwriting Canvas & Document Synthesis</h2>
        <p className="text-xs text-slate-400 mt-1">
          Translate the Character-Level RNN's generated sequence into an authentic handwritten letter with customizable paper, ink shades, and organic stroke jitter.
        </p>
      </div>

      {/* Synthesis Controls Panel */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Sliders className="w-4 h-4 text-indigo-400" />
          <span>Calligraphy & Paper Customization</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          
          {/* Pen / Font Style */}
          <div>
            <label className="text-slate-400 block mb-1">Handwriting Stroke Style:</label>
            <select
              value={handwritingStyle}
              onChange={(e) => setHandwritingStyle(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="font-handwriting-caveat">Fountain Pen Cursive (Caveat)</option>
              <option value="font-handwriting-apple">Vintage Manuscript (Homemade Apple)</option>
              <option value="font-handwriting-cedar">Casual Ballpoint (Cedarville)</option>
              <option value="font-handwriting-shadows">Architect / Clean Script (Shadows)</option>
              <option value="font-handwriting-reenie">Loose Diary Scribble (Reenie)</option>
            </select>
          </div>

          {/* Paper Background */}
          <div>
            <label className="text-slate-400 block mb-1">Paper Surface Texture:</label>
            <select
              value={paperStyle}
              onChange={(e) => setPaperStyle(e.target.value as any)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="lined">Lined College Journal (Margin Rule)</option>
              <option value="grid">Engineering Graph Paper</option>
              <option value="parchment">Clean Cotton Parchment</option>
              <option value="chalkboard">Dark Slate Chalkboard</option>
            </select>
          </div>

          {/* Ink Color */}
          <div>
            <label className="text-slate-400 block mb-1">Fountain Pen Ink:</label>
            <div className="flex items-center gap-2 mt-1">
              {inkPalette.map((ink) => (
                <button
                  key={ink.value}
                  onClick={() => setInkColor(ink.value)}
                  style={{ backgroundColor: ink.value }}
                  className={`w-6 h-6 rounded-full border-2 transition-transform cursor-pointer ${
                    inkColor === ink.value ? 'scale-110 border-white shadow-sm' : 'border-transparent hover:scale-105'
                  }`}
                  title={ink.name}
                />
              ))}
            </div>
          </div>

          {/* Font Size & Slant */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Scale:</span>
              <span className="text-indigo-400 font-mono-code font-bold">{fontSize}px</span>
            </div>
            <input
              type="range"
              min={18}
              max={36}
              value={fontSize}
              onChange={(e) => {
                setFontSize(Number(e.target.value));
                setLineHeight(Math.round(Number(e.target.value) * 1.3));
              }}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>

        </div>

        {/* Action Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800">
          <div className="text-[11px] text-slate-400">
            Edit text in the canvas below directly or generate new text in the "Generation" tab.
          </div>

          <button
            onClick={handleExportPng}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors shadow-sm cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Document as High-Res PNG</span>
          </button>
        </div>
      </div>

      {/* Interactive Paper Canvas View */}
      <div className="rounded-2xl border border-slate-700/80 p-2 sm:p-6 bg-slate-950/40">
        <div
          className={`w-full min-h-[420px] rounded-xl p-8 sm:p-12 transition-all ${paperClasses[paperStyle]}`}
          style={{
            transform: `rotate(${slantAngle * 0.15}deg)`,
          }}
        >
          <div
            contentEditable
            suppressContentEditableWarning
            onBlur={(e) => setText(e.currentTarget.innerText)}
            className={`w-full outline-none whitespace-pre-wrap ${handwritingStyle}`}
            style={{
              color: inkColor,
              fontSize: `${fontSize}px`,
              lineHeight: `${lineHeight}px`,
              letterSpacing: `${letterSpacing}px`,
            }}
          >
            {text}
          </div>
        </div>
      </div>

    </div>
  );
};
