import React from 'react';
import { PredictResponse, CANONICAL_EXPRESSIONS, EXPRESSION_EMOJIS, ExpressionType } from '../services/api';
import { CheckCircle2, AlertTriangle, HelpCircle, Compass, ShieldCheck } from 'lucide-react';

interface PredictionPanelProps {
  prediction: PredictResponse | null;
  cameraActive: boolean;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  prediction,
  cameraActive,
}) => {
  const isBackendDown = prediction ? (prediction.backend_connected === false || prediction.status === 'UNAVAILABLE' || prediction.prediction === 'UNAVAILABLE') : false;
  const hasFace = !isBackendDown && cameraActive && Boolean(prediction?.face_detected);
  const dominantExpr = isBackendDown ? 'UNAVAILABLE' : hasFace && prediction ? prediction.prediction : '—';
  const dominantConf = !isBackendDown && hasFace && prediction ? (prediction.confidence * 100).toFixed(1) : '0.0';
  const emoji = !isBackendDown && hasFace ? EXPRESSION_EMOJIS[dominantExpr] || '' : '';

  const pitch = !isBackendDown && hasFace && prediction?.head_pose ? prediction.head_pose.pitch : 0;
  const yaw = !isBackendDown && hasFace && prediction?.head_pose ? prediction.head_pose.yaw : 0;
  const roll = !isBackendDown && hasFace && prediction?.head_pose ? prediction.head_pose.roll : 0;

  const formatAngle = (val: number) => {
    return val > 0 ? `+${val.toFixed(1)}°` : `${val.toFixed(1)}°`;
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Backend Unavailable Alert Banner */}
      {isBackendDown && (
        <div className="p-3.5 rounded-xl bg-red-950/70 border border-red-500/50 text-red-200 flex items-start gap-3 shadow-lg">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <div className="font-semibold text-red-100 flex items-center gap-1.5 font-mono uppercase tracking-wide">
              <span>PREDICTION UNAVAILABLE</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-900/80 border border-red-700/60 text-red-300">
                Backend Connection Lost
              </span>
            </div>
            <p className="text-red-300/90 leading-relaxed font-sans">
              The FastAPI ML inference backend is unreachable. Handcrafted, brightness-based, and client fallback predictions are strictly disabled in accordance with model integrity standards.
            </p>
          </div>
        </div>
      )}

      {/* 1. Hero Dominant Prediction Card */}
      <div className={`p-4 sm:p-5 rounded-2xl bg-slate-900/90 border backdrop-blur-xl shadow-xl flex flex-col justify-between relative overflow-hidden ${
        isBackendDown ? 'border-red-900/40' : 'border-slate-800'
      }`}>
        {/* Ambient cyan or red glow */}
        <div className={`absolute -top-10 -right-10 w-36 h-36 rounded-full blur-2xl pointer-events-none ${
          isBackendDown ? 'bg-red-500/10' : 'bg-cyan-500/10'
        }`} />

        <div className="flex items-center justify-between pb-2.5 border-b border-slate-800/80">
          <span className={`font-mono text-xs font-semibold tracking-wider uppercase ${
            isBackendDown ? 'text-red-400' : 'text-cyan-400'
          }`}>
            PRIMARY EXPRESSION
          </span>

          {isBackendDown ? (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-red-950/80 text-red-400 border border-red-700/60">
              <AlertTriangle className="w-3 h-3" />
              OFFLINE
            </span>
          ) : hasFace ? (
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

        {/* Hero Expression & Confidence Display */}
        <div className="my-3 flex items-baseline justify-between gap-2">
          <div className="min-w-0">
            <h2 className={`text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2 sm:gap-3 truncate ${
              isBackendDown ? 'text-red-300' : 'text-white'
            }`}>
              <span className="truncate">{dominantExpr.toUpperCase()}</span>
              {emoji && <span className="text-3xl sm:text-4xl shrink-0">{emoji}</span>}
            </h2>
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mt-1">
              MODEL CONFIDENCE
            </span>
          </div>

          <div className="text-right shrink-0">
            <span className={`font-mono text-3xl sm:text-4xl font-extrabold tracking-tight ${
              isBackendDown ? 'text-slate-600' : 'text-cyan-400'
            }`}>
              {dominantConf}%
            </span>
          </div>
        </div>

        {/* Primary Facial Attributes Grid */}
        <div className="pt-2.5 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
          {/* Geometry status */}
          <div className="p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
            <span className="text-slate-400 text-[9px] block uppercase">GEOMETRY</span>
            {isBackendDown ? (
              <span className="text-slate-600 font-semibold">OFFLINE</span>
            ) : hasFace && prediction?.geometry_valid ? (
              <span className="text-emerald-400 font-semibold">VALID (62-D)</span>
            ) : hasFace && prediction?.partial_face ? (
              <span className="text-amber-400 font-semibold">DEGRADED</span>
            ) : (
              <span className="text-slate-500">AWAITING</span>
            )}
          </div>

          {/* Detection Confidence */}
          <div className="p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
            <span className="text-slate-400 text-[9px] block uppercase">DETECTION CONF</span>
            <span className="text-slate-200 font-semibold">
              {!isBackendDown && hasFace && prediction ? `${(prediction.detection_confidence * 100).toFixed(0)}%` : '—'}
            </span>
          </div>

          {/* Head Pose */}
          <div className="col-span-2 sm:col-span-1 p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
            <span className="text-slate-400 text-[9px] block uppercase">POSE (P/Y/R)</span>
            <span className="text-slate-300 truncate block">
              {!isBackendDown && hasFace ? `${formatAngle(pitch)} / ${formatAngle(yaw)}` : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Seven Class Probabilities Card */}
      <div className="p-4 sm:p-5 rounded-2xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl shadow-xl flex flex-col gap-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
          <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase">
            7-CLASS PROBABILITY DISTRIBUTION
          </span>
          <span className="text-[10px] font-mono text-slate-400">SOFTMAX POSTERIORS</span>
        </div>

        <div className="space-y-2.5">
          {CANONICAL_EXPRESSIONS.map((expr) => {
            const rawProb = !isBackendDown && hasFace && prediction?.probabilities ? (prediction.probabilities[expr] || 0) : 0;
            const percent = (rawProb * 100).toFixed(1);
            const isDominant = !isBackendDown && hasFace && prediction?.prediction === expr;

            return (
              <div key={expr} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span
                    className={`font-medium flex items-center gap-2 ${
                      isDominant ? 'text-cyan-300 font-bold' : isBackendDown ? 'text-slate-500' : 'text-slate-300'
                    }`}
                  >
                    <span className={`text-sm shrink-0 ${isBackendDown ? 'opacity-40 grayscale' : ''}`}>
                      {EXPRESSION_EMOJIS[expr]}
                    </span>
                    <span>{expr}</span>
                  </span>
                  <span
                    className={`font-mono text-xs ${
                      isDominant ? 'text-cyan-400 font-bold' : isBackendDown ? 'text-slate-600' : 'text-slate-400'
                    }`}
                  >
                    {percent}%
                  </span>
                </div>

                {/* Horizontal Probability Gauge */}
                <div className="w-full h-1.5 rounded-full bg-slate-950 border border-slate-800/80 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ease-out ${
                      isDominant
                        ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.7)]'
                        : isBackendDown
                        ? 'bg-slate-900'
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
