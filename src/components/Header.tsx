import React from 'react';
import { Download, Sparkles, BookOpen, Layers, Play, PenTool, Database, Code2, CheckCircle2 } from 'lucide-react';
import { downloadPythonProjectZip } from '../utils/zipGenerator';

export type TabType = 
  | 'overview'
  | 'dataset'
  | 'training'
  | 'generation'
  | 'handwriting'
  | 'kaggle'
  | 'code'
  | 'submission';

interface HeaderProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  isModelTrained: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, isModelTrained }) => {
  const [isDownloading, setIsDownloading] = React.useState(false);

  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      await downloadPythonProjectZip();
    } finally {
      setIsDownloading(false);
    }
  };

  const navItems: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <BookOpen className="w-3.5 h-3.5" /> },
    { id: 'dataset', label: 'Dataset', icon: <Database className="w-3.5 h-3.5" /> },
    { id: 'training', label: 'Training', icon: <Layers className="w-3.5 h-3.5" /> },
    { id: 'generation', label: 'Generation', icon: <Play className="w-3.5 h-3.5" /> },
    { id: 'handwriting', label: 'Handwriting Canvas', icon: <PenTool className="w-3.5 h-3.5" /> },
    { id: 'kaggle', label: 'Kaggle Dataset', icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: 'code', label: 'Python & Colab', icon: <Code2 className="w-3.5 h-3.5" /> },
    { id: 'submission', label: 'Submission Guide', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className="sticky top-0 z-40 bg-slate-900/95 backdrop-blur border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
        
        {/* Zone 1: Brand Wordmark (Single text element) */}
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-sm font-bold text-lg">
            ✍️
          </div>
          <div>
            <span className="text-base font-bold tracking-tight text-white whitespace-nowrap block">
              Handwriting Studio
            </span>
            <span className="text-[11px] text-slate-400 whitespace-nowrap block">
              Character-Level RNN · CODSOFT Task 2
            </span>
          </div>
        </div>

        {/* Zone 2: Navigation Links (Single row, clean typography, no pill boxes) */}
        <nav className="hidden lg:flex items-center gap-1 xl:gap-2 text-xs font-medium overflow-x-auto">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  isActive
                    ? 'bg-slate-800 text-indigo-400 font-semibold'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Zone 3: Primary Action Button */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={handleDownload}
            disabled={isDownloading}
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 rounded-lg transition-colors shadow-sm disabled:opacity-60 whitespace-nowrap cursor-pointer"
            title="Download full runnable Python codebase with requirements.txt, sample datasets, and Colab guide"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isDownloading ? 'Packaging...' : 'Download Project (.zip)'}</span>
          </button>
        </div>

      </div>

      {/* Mobile Navigation bar */}
      <div className="lg:hidden flex items-center gap-1 px-4 py-2 border-t border-slate-800/80 overflow-x-auto bg-slate-950/60">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-1 px-2.5 py-1 text-[11px] rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                isActive
                  ? 'bg-indigo-600/30 text-indigo-300 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </header>
  );
};
