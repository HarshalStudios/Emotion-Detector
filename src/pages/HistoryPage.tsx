import React from 'react';
import { History, Trash2, ShieldCheck } from 'lucide-react';
import { EXPRESSION_EMOJIS } from '../services/api';

export interface HistoryRecord {
  id: string;
  timestamp: string;
  prediction: string;
  confidence: number;
  latencyMs: number;
  geometryValid: boolean;
}

interface HistoryPageProps {
  history: HistoryRecord[];
  onClearHistory: () => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  history,
  onClearHistory,
}) => {
  return (
    <div className="space-y-5 sm:space-y-6 py-4 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-slate-800">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-xs font-mono font-medium">
            SESSION TELEMETRY LOG
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1.5">
            Inference History Log
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Real-time inference records captured during active webcam and sample evaluation sessions.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="min-h-[40px] px-3.5 py-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-rose-400 text-xs font-mono transition-colors flex items-center gap-1.5 self-start sm:self-auto active:scale-95"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Session Log</span>
          </button>
        )}
      </div>

      {/* History Records Container */}
      <div className="rounded-2xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl shadow-xl overflow-hidden">
        {history.length === 0 ? (
          <div className="p-8 sm:p-12 text-center space-y-3">
            <History className="w-8 h-8 text-slate-600 mx-auto" />
            <h3 className="text-sm font-semibold text-slate-300">No Inferences Logged Yet</h3>
            <p className="text-xs text-slate-500 max-w-xs mx-auto">
              Start live analysis or test benchmark sample images in the Workstation to log real-time expression inferences.
            </p>
          </div>
        ) : (
          <>
            {/* Desktop / Tablet Table View (hidden on narrow screens < 640px) */}
            <div className="hidden sm:block overflow-x-auto font-mono text-xs">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase bg-slate-950/80 tracking-wider">
                    <th className="py-2.5 px-4">TIMESTAMP</th>
                    <th className="py-2.5 px-4">PREDICTION</th>
                    <th className="py-2.5 px-4">CONFIDENCE</th>
                    <th className="py-2.5 px-4">GEOMETRY</th>
                    <th className="py-2.5 px-4">LATENCY</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {history.map((rec) => (
                    <tr key={rec.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-2.5 px-4 text-slate-400">{rec.timestamp}</td>
                      <td className="py-2.5 px-4 font-semibold text-white">
                        <div className="flex items-center gap-2">
                          <span className="text-base">{EXPRESSION_EMOJIS[rec.prediction] || '😐'}</span>
                          <span className="tracking-wide">{rec.prediction.toUpperCase()}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-4 text-cyan-400 font-bold">
                        {(rec.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4">
                        {rec.geometryValid ? (
                          <span className="text-emerald-400 font-semibold">VALID (62-D)</span>
                        ) : (
                          <span className="text-slate-500">STANDBY</span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 text-slate-300">{rec.latencyMs.toFixed(0)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile Responsive Cards View (visible on < 640px) */}
            <div className="sm:hidden divide-y divide-slate-800/80 p-2 font-mono">
              {history.map((rec) => (
                <div key={rec.id} className="p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{EXPRESSION_EMOJIS[rec.prediction] || '😐'}</span>
                      <span className="text-sm font-bold text-white">{rec.prediction.toUpperCase()}</span>
                    </div>
                    <span className="text-sm font-bold text-cyan-400">
                      {(rec.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
                    <span>{rec.timestamp}</span>
                    <div className="flex items-center gap-3">
                      <span>{rec.latencyMs.toFixed(0)} ms</span>
                      {rec.geometryValid ? (
                        <span className="text-emerald-400 font-semibold">62-D OK</span>
                      ) : (
                        <span className="text-slate-500">STANDBY</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
