import React from 'react';
import { History, Trash2 } from 'lucide-react';
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
    <div className="space-y-6 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-xs font-mono font-medium">
            RESEARCH LOG
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight mt-2">
            Inference Session History
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Ephemeral prediction stream captured during active live analysis sessions.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={onClearHistory}
            className="px-3.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-rose-400 text-xs font-mono transition-colors flex items-center gap-1.5 self-start sm:self-auto"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Log</span>
          </button>
        )}
      </div>

      {/* History Table: Clean Research Log Style */}
      <div className="rounded-2xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl shadow-xl overflow-hidden">
        {history.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <History className="w-8 h-8 text-slate-600 mx-auto" />
            <h3 className="text-sm font-semibold text-slate-300">No Inferences Logged Yet</h3>
            <p className="text-xs text-slate-500 max-w-xs mx-auto">
              Start live analysis in the Workstation to capture real-time expression inferences.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto font-mono text-xs">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase bg-slate-950/80 tracking-wider">
                  <th className="py-2.5 px-4">TIME</th>
                  <th className="py-2.5 px-4">EXPRESSION</th>
                  <th className="py-2.5 px-4">CONFIDENCE</th>
                  <th className="py-2.5 px-4">GEOMETRY</th>
                  <th className="py-2.5 px-4">LATENCY</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {history.map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-4 text-slate-400">{rec.timestamp}</td>
                    <td className="py-2.5 px-4 font-semibold text-white flex items-center gap-2">
                      <span>{EXPRESSION_EMOJIS[rec.prediction] || '😐'}</span>
                      <span className="tracking-wide">{rec.prediction.toUpperCase()}</span>
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
        )}
      </div>
    </div>
  );
};
