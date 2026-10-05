import React from 'react';
import { Camera, Scan, Crop, Layers, GitMerge, CheckCircle } from 'lucide-react';

interface PipelineBarProps {
  currentStep?: number;
}

export const PipelineBar: React.FC<PipelineBarProps> = ({ currentStep = 6 }) => {
  const steps = [
    { num: '01', title: 'CAMERA', desc: 'Single RGB Stream', icon: Camera },
    { num: '02', title: 'DETECTION', desc: 'Face Mesh Localization', icon: Scan },
    { num: '03', title: 'PREPROCESS', desc: '5-Pt Align · 224×224', icon: Crop },
    { num: '04', title: 'REPRESENTATIONS', desc: 'Spatial + FFT + 62-D', icon: Layers },
    { num: '05', title: 'FUSION', desc: 'A4 Concatenated MLP', icon: GitMerge },
    { num: '06', title: 'PREDICTION', desc: '7-Class RAF-DB ONNX', icon: CheckCircle },
  ];

  return (
    <div className="w-full bg-slate-900/70 border border-slate-800 rounded-xl p-4 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase">
            ACTIVE PIPELINE ARCHITECTURE
          </span>
          <span className="text-[10px] font-mono text-slate-400">· A4 RESEARCH SPECIFICATION</span>
        </div>
        <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          END-TO-END VERIFIED
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {steps.map((s, idx) => {
          const Icon = s.icon;
          const isActive = idx + 1 <= currentStep;
          return (
            <div
              key={s.num}
              className={`p-2.5 rounded-lg border transition-all ${
                isActive
                  ? 'bg-slate-950/80 border-slate-800 text-slate-200'
                  : 'bg-slate-950/40 border-slate-900 text-slate-500'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-[10px] text-cyan-400 font-bold">{s.num}</span>
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-600'}`} />
              </div>
              <h4 className="font-mono text-xs font-bold text-slate-100 tracking-tight">{s.title}</h4>
              <p className="text-[10px] text-slate-400 truncate mt-0.5">{s.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
