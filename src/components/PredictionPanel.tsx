import React from 'react';
import { PredictResponse, CANONICAL_EXPRESSIONS, EXPRESSION_EMOJIS } from '../services/api';
import { CheckCircle2, AlertTriangle, HelpCircle } from 'lucide-react';

interface PredictionPanelProps {
  prediction: PredictResponse | null;
  cameraActive: boolean;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  prediction,
  cameraActive,
}) => {
  const hasFace = cameraActive && prediction?.face_detected;
  const dominantExpr = hasFace ? prediction.prediction : '—';
  const dominantConf = hasFace ? (prediction.confidence * 100).toFixed(1) : '0.0';
  const emoji = hasFace ? EXPRESSION_EMOJIS[dominantExpr] || '' : '';

  return (
    <div className="flex flex-col gap-4">
      {/* 1. Hero Dominant Prediction Card */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl shadow-xl flex flex-col justify-between relative overflow-hidden">
        {/* Subtle radial cyan tint behind prediction */}
        <div className="absolute -top-10 -right-10 w-36 h-36 bg-cyan-500/10 rounded-full blur-2xl pointer-events-none" />

        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
          <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase">
            CURRENT EXPRESSION
          </span>

          {hasFace ? (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/70 text-emerald-400 border border-emerald-600/40">
              <CheckCircle2 className="w-3 h-3" />
              CONFIRMED
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-800 text-slate-400">
              <HelpCircle className="w-3 h-3" />
              STANDBY
            </span>
          )}
        </div>

        <div className="my-3 flex items-baseline justify-between">
          <div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
              <span>{dominantExpr.toUpperCase()}</span>
              {emoji && <span className="text-3xl sm:text-4xl">{emoji}</span>}
            </h2>
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mt-1">
              MODEL CONFIDENCE
            </span>
          </div>

          <div className="text-right">
            <span className="font-mono text-3xl sm:text-4xl font-extrabold text-cyan-400 tracking-tight">
              {dominantConf}%
            </span>
          </div>
        </div>

        {/* State Status Indicator Line */}
        <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
          <span className="text-slate-400">GEOMETRY:</span>
          {hasFace && prediction?.geometry_valid ? (
            <span className="text-emerald-400 font-semibold">VALID (62-D)</span>
          ) : hasFace && prediction?.partial_face ? (
            <span className="text-amber-400 font-semibold">PARTIAL DEGRADED</span>
          ) : (
            <span className="text-slate-500">AWAITING FACE</span>
          )}
        </div>
      </div>

      {/* 2. Seven Class Probabilities Card */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl shadow-xl flex flex-col gap-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
          <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase">
            7-CLASS PROBABILITY DISTRIBUTION
          </span>
          <span className="text-[10px] font-mono text-slate-400">RAF-DB POSTERIORS</span>
        </div>

        <div className="space-y-2.5">
          {CANONICAL_EXPRESSIONS.map((expr) => {
            const rawProb = hasFace && prediction.probabilities ? (prediction.probabilities[expr] || 0) : 0;
            const percent = (rawProb * 100).toFixed(1);
            const isDominant = hasFace && prediction.prediction === expr;

            return (
              <div key={expr} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span
                    className={`font-medium flex items-center gap-2 ${
                      isDominant ? 'text-cyan-300 font-bold' : 'text-slate-300'
                    }`}
                  >
                    <span className="text-sm">{EXPRESSION_EMOJIS[expr]}</span>
                    <span>{expr}</span>
                  </span>
                  <span
                    className={`font-mono text-xs ${
                      isDominant ? 'text-cyan-400 font-bold' : 'text-slate-400'
                    }`}
                  >
                    {percent}%
                  </span>
                </div>

                {/* Clean Horizontal Probability Gauge */}
                <div className="w-full h-1.5 rounded-full bg-slate-950 border border-slate-800/80 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ease-out ${
                      isDominant
                        ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.7)]'
                        : 'bg-slate-700/60'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, parseFloat(percent)))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
