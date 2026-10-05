import React from 'react';
import { Scan, ShieldCheck, Database, Cpu } from 'lucide-react';
import { NavTab } from './Header';

interface FooterProps {
  onNavigate: (tab: NavTab) => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="mt-16 border-t border-slate-800/80 bg-slate-950/80 text-xs text-slate-500 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Scan className="w-3.5 h-3.5" />
            </div>
            <span className="font-mono text-xs font-bold text-slate-300">
              EMOTION DETECTOR // REAL-TIME FACIAL EXPRESSION ANALYSIS
            </span>
          </div>

          <div className="flex items-center gap-6 font-mono text-[11px] text-slate-400">
            <button onClick={() => onNavigate('home')} className="hover:text-cyan-400 transition-colors">
              Home
            </button>
            <button onClick={() => onNavigate('analyze')} className="hover:text-cyan-400 transition-colors">
              Analyze
            </button>
            <button onClick={() => onNavigate('how-it-works')} className="hover:text-cyan-400 transition-colors">
              How It Works
            </button>
            <button onClick={() => onNavigate('research')} className="hover:text-cyan-400 transition-colors">
              Research
            </button>
            <button onClick={() => onNavigate('history')} className="hover:text-cyan-400 transition-colors">
              History
            </button>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-900 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-[11px] text-slate-400">
          <p className="max-w-2xl">
            Candidate A4: Spatial (ConvNeXt-Tiny) + Frequency (2D-FFT) + Geometry (MediaPipe 62-D) Multi-Representation Fusion. Analyzes visible facial Action Unit muscle activations and physical expressions; does not claim to infer internal subjective emotional states.
          </p>
          <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400 shrink-0">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>FASTAPI POST /predict COMPLIANT</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
