import React from 'react';
import { PredictResponse, CANONICAL_EXPRESSIONS, EXPRESSION_EMOJIS, ExpressionType } from '../services/api';
import { CheckCircle2, AlertTriangle, HelpCircle, RefreshCw, XCircle } from 'lucide-react';

interface PredictionPanelProps {
  prediction: PredictResponse | null;
  cameraActive: boolean;
  mode?: 'webcam' | 'sample';
  referenceEmotion?: ExpressionType | null;
  isEvaluating?: boolean;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  prediction,
  cameraActive,
  mode = 'webcam',
  referenceEmotion = null,
  isEvaluating = false,
}) => {
  const isBackendDown = prediction
    ? prediction.backend_connected === false ||
      prediction.status === 'UNAVAILABLE' ||
      prediction.prediction === 'UNAVAILABLE'
    : false;

  const hasFace = !isBackendDown && cameraActive && Boolean(prediction?.face_detected);
  const dominantExpr = isBackendDown ? 'UNAVAILABLE' : hasFace && prediction ? prediction.prediction : '—';
  const dominantConf = !isBackendDown && hasFace && prediction ? (prediction.confidence * 100).toFixed(1) : '0.0';
  const modelEmoji = !isBackendDown && hasFace ? EXPRESSION_EMOJIS[dominantExpr] || '' : '';

  const pitch = !isBackendDown && hasFace && prediction?.head_pose ? prediction.head_pose.pitch : 0;
  const yaw = !isBackendDown && hasFace && prediction?.head_pose ? prediction.head_pose.yaw : 0;

  const formatAngle = (val: number) => {
    return val > 0 ? `+${val.toFixed(1)}°` : `${val.toFixed(1)}°`;
  };

  const isSampleMode = mode === 'sample';
  const refEmoji = referenceEmotion ? EXPRESSION_EMOJIS[referenceEmotion] || '' : '';

  // Scientific evaluation: true only when backend predicted class equals ground-truth reference
  const isMatch =
    isSampleMode &&
    referenceEmotion &&
    hasFace &&
    !isBackendDown &&
    prediction &&
    dominantExpr.toLowerCase() === referenceEmotion.toLowerCase();

  const isDisagreement =
    isSampleMode &&
    referenceEmotion &&
    hasFace &&
    !isBackendDown &&
    prediction &&
    dominantExpr.toLowerCase() !== referenceEmotion.toLowerCase();

  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* Backend Unavailable Alert */}
      {isBackendDown && (
        <div className="p-3.5 rounded-xl bg-rose-950/80 border border-rose-500/40 text-rose-200 flex items-start gap-3 shadow-md">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <div className="font-semibold text-rose-100 flex items-center gap-1.5 font-mono uppercase tracking-wide">
              <span>PREDICTION BACKEND OFFLINE</span>
            </div>
            <p className="text-rose-300/90 leading-relaxed font-sans">
              The FastAPI ML inference backend is unreachable. Real model evaluation is paused.
            </p>
          </div>
        </div>
      )}

      {/* SAMPLE MODE: REFERENCE vs MODEL OUTPUT vs EVALUATION */}
      {isSampleMode && referenceEmotion ? (
        <div className="rounded-xl bg-slate-900 border border-slate-800 shadow-md overflow-hidden">
          {/* Header Bar */}
          <div className="px-4 py-2.5 bg-slate-950/70 border-b border-slate-800 flex items-center justify-between font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="font-semibold text-slate-200 tracking-wider uppercase text-[11px]">
                SAMPLE BENCHMARK EVALUATION
              </span>
            </div>
            {isEvaluating ? (
              <span className="inline-flex items-center gap-1 text-[11px] text-cyan-400 font-medium">
                <RefreshCw className="w-3 h-3 animate-spin" />
                RUNNING INFERENCE...
              </span>
            ) : (
              <span className="text-[10px] text-slate-400 uppercase">
                A4 TRIPLE-REPRESENTATION PIPELINE
              </span>
            )}
          </div>

          <div className="p-4 space-y-3.5">
            {/* Split Comparison Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* 1. Reference Expression (Ground Truth) */}
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80 flex flex-col justify-between">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/60">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider font-semibold">
                    REFERENCE EXPRESSION
                  </span>
                  <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    GROUND TRUTH
                  </span>
                </div>
                <div className="my-2.5 flex items-center gap-2.5">
                  <span className="text-3xl shrink-0">{refEmoji}</span>
                  <div className="min-w-0">
                    <span className="text-xl sm:text-2xl font-bold tracking-tight text-white block truncate uppercase">
                      {referenceEmotion}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 block">
                      Labeled Dataset Target
                    </span>
                  </div>
                </div>
              </div>

              {/* 2. Model Output (Actual Prediction) */}
              <div className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80 flex flex-col justify-between">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/60">
                  <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold">
                    A4 MODEL OUTPUT
                  </span>
                  <span className="text-[10px] font-mono font-bold text-cyan-300">
                    {dominantConf}% CONF
                  </span>
                </div>
                <div className="my-2.5 flex items-center gap-2.5">
                  <span className="text-3xl shrink-0">{modelEmoji || '—'}</span>
                  <div className="min-w-0">
                    <span className={`text-xl sm:text-2xl font-bold tracking-tight block truncate uppercase ${
                      isBackendDown ? 'text-rose-400' : 'text-cyan-300'
                    }`}>
                      {dominantExpr}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 block">
                      Authoritative ML Output
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* 3. Match / Disagreement Evaluation Banner */}
            <div className="pt-0.5">
              {isEvaluating ? (
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2.5 text-xs font-mono text-slate-400">
                  <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
                  <span>Streaming tensor to A4 model forward pass...</span>
                </div>
              ) : isMatch ? (
                <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-500/40 flex items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2.5">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                    <div>
                      <span className="font-mono font-bold text-emerald-300 text-sm tracking-wide block">
                        ✓ MATCH
                      </span>
                      <span className="text-[11px] text-emerald-200/80 font-sans">
                        A4 model prediction corresponds to reference label ({referenceEmotion}).
                      </span>
                    </div>
                  </div>
                  <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-emerald-900/60 text-emerald-300 border border-emerald-600/40 shrink-0 font-medium">
                    VERIFIED
                  </span>
                </div>
              ) : isDisagreement ? (
                <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-500/40 flex items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2.5">
                    <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
                    <div>
                      <span className="font-mono font-bold text-amber-300 text-sm tracking-wide block">
                        ⚠ MODEL DISAGREEMENT
                      </span>
                      <span className="text-[11px] text-amber-200/80 font-sans">
                        Model predicted <strong className="text-amber-100">{dominantExpr}</strong> ({dominantConf}%) vs labeled reference <strong className="text-amber-100">{referenceEmotion}</strong>. Genuine model variance.
                      </span>
                    </div>
                  </div>
                  <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-amber-900/60 text-amber-300 border border-amber-600/40 shrink-0 font-medium">
                    DISCREPANCY
                  </span>
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2.5 text-xs font-mono text-slate-400">
                  <HelpCircle className="w-4 h-4 text-slate-500 shrink-0" />
                  <span>Select a reference sample to inspect live evaluation.</span>
                </div>
              )}
            </div>

            {/* Geometry & Pose Telemetry Row */}
            <div className="pt-2 border-t border-slate-800/80 grid grid-cols-3 gap-2 font-mono text-[11px]">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 text-[9px] block uppercase">GEOMETRY (FACS)</span>
                {prediction?.geometry_valid ? (
                  <span className="text-emerald-400 font-semibold">VALID (62-D)</span>
                ) : (
                  <span className="text-slate-500">STANDBY</span>
                )}
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 text-[9px] block uppercase">DETECTION CONF</span>
                <span className="text-slate-200 font-semibold">
                  {prediction ? `${(prediction.detection_confidence * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <span className="text-slate-400 text-[9px] block uppercase">HEAD POSE (P/Y)</span>
                <span className="text-slate-300 truncate block">
                  {hasFace ? `${formatAngle(pitch)} / ${formatAngle(yaw)}` : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* LIVE WEBCAM MODE: NO REFERENCE LABEL, MODEL OUTPUT ONLY */
        <div className={`p-4 sm:p-5 rounded-xl bg-slate-900 border backdrop-blur-xl shadow-md flex flex-col justify-between relative ${
          isBackendDown ? 'border-rose-900/50' : 'border-slate-800'
        }`}>
          <div className="flex items-center justify-between pb-2.5 border-b border-slate-800/80">
            <span className={`font-mono text-xs font-semibold tracking-wider uppercase ${
              isBackendDown ? 'text-rose-400' : 'text-cyan-400'
            }`}>
              PRIMARY EXPRESSION
            </span>

            {isBackendDown ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-rose-950 text-rose-400 border border-rose-700/60">
                <AlertTriangle className="w-3 h-3" />
                OFFLINE
              </span>
            ) : hasFace ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950 text-emerald-400 border border-emerald-600/40">
                <CheckCircle2 className="w-3 h-3" />
                DETECTED
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-800 text-slate-400">
                <HelpCircle className="w-3 h-3" />
                STANDBY
              </span>
            )}
          </div>

          {/* Hero Expression Display */}
          <div className="my-3 flex items-baseline justify-between gap-2">
            <div className="min-w-0">
              <h2 className={`text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5 truncate ${
                isBackendDown ? 'text-rose-300' : 'text-white'
              }`}>
                <span className="truncate uppercase">{dominantExpr}</span>
                {modelEmoji && <span className="text-3xl sm:text-4xl shrink-0">{modelEmoji}</span>}
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

          {/* Telemetry Row */}
          <div className="pt-2.5 border-t border-slate-800/80 grid grid-cols-3 gap-2 font-mono text-[11px]">
            <div className="p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
              <span className="text-slate-400 text-[9px] block uppercase">GEOMETRY</span>
              {prediction?.geometry_valid ? (
                <span className="text-emerald-400 font-semibold">VALID (62-D)</span>
              ) : (
                <span className="text-slate-500">STANDBY</span>
              )}
            </div>

            <div className="p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
              <span className="text-slate-400 text-[9px] block uppercase">DETECTION CONF</span>
              <span className="text-slate-200 font-semibold">
                {hasFace && prediction ? `${(prediction.detection_confidence * 100).toFixed(0)}%` : '—'}
              </span>
            </div>

            <div className="p-1.5 rounded bg-slate-950/60 border border-slate-800/60">
              <span className="text-slate-400 text-[9px] block uppercase">POSE (P/Y)</span>
              <span className="text-slate-300 truncate block">
                {hasFace ? `${formatAngle(pitch)} / ${formatAngle(yaw)}` : '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 7-Class Probability Distribution */}
      <div className="p-4 sm:p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-md flex flex-col gap-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase">
            7-CLASS PROBABILITY DISTRIBUTION
          </span>
          <span className="text-[10px] font-mono text-slate-400">SOFTMAX POSTERIORS</span>
        </div>

        <div className="space-y-2.5">
          {CANONICAL_EXPRESSIONS.map((expr) => {
            const rawProb = !isBackendDown && hasFace && prediction?.probabilities ? (prediction.probabilities[expr] || 0) : 0;
            const percent = (rawProb * 100).toFixed(1);
            const isPredicted = !isBackendDown && hasFace && prediction?.prediction === expr;
            const isReference = isSampleMode && referenceEmotion === expr;

            return (
              <div key={expr} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className={`font-medium flex items-center gap-1.5 ${
                    isPredicted
                      ? 'text-cyan-300 font-bold'
                      : isReference
                      ? 'text-slate-200 font-semibold'
                      : isBackendDown
                      ? 'text-slate-500'
                      : 'text-slate-300'
                  }`}>
                    <span>{EXPRESSION_EMOJIS[expr]}</span>
                    <span>{expr}</span>
                    {isReference && (
                      <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        REF
                      </span>
                    )}
                    {isPredicted && (
                      <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                        PRED
                      </span>
                    )}
                  </span>
                  <span className="font-mono text-slate-300 text-xs">{percent}%</span>
                </div>

                <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800/80">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      isPredicted
                        ? 'bg-cyan-400'
                        : isReference
                        ? 'bg-slate-400'
                        : 'bg-slate-700'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, rawProb * 100))}%` }}
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
