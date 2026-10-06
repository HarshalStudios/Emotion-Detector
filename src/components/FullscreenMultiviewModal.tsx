import React from 'react';
import { X, Sparkles, CheckCircle2, ShieldCheck, Activity } from 'lucide-react';
import { PredictResponse, CANONICAL_EXPRESSIONS, EXPRESSION_EMOJIS } from '../services/api';

interface FullscreenMultiviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  prediction: PredictResponse | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  selectedSource: 'webcam' | 'sample';
  activeSampleImage?: string;
  fps: number;
  latencyMs: number;
}

export const FullscreenMultiviewModal: React.FC<FullscreenMultiviewModalProps> = ({
  isOpen,
  onClose,
  prediction,
  videoRef,
  selectedSource,
  activeSampleImage = '/samples/neutral/neutral_01.jpg',
  fps,
  latencyMs,
}) => {
  if (!isOpen) return null;

  const hasFace = prediction?.face_detected;
  const dominantExpr = hasFace ? prediction.prediction : 'Searching Face...';
  const dominantConf = hasFace ? (prediction.confidence * 100).toFixed(1) : '0.0';
  const emoji = hasFace ? EXPRESSION_EMOJIS[dominantExpr] || '😐' : '👤';

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/98 backdrop-blur-2xl flex flex-col p-3 sm:p-6 overflow-y-auto animate-in fade-in duration-200 w-full">
      {/* Top Header Bar */}
      <div className="flex items-center justify-between pb-3 sm:pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 font-mono text-xs font-bold">
            COMMAND CENTER
          </div>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            A4 REAL-TIME FACIAL ANALYSIS WORKSTATION
          </span>
        </div>

        <button
          onClick={onClose}
          className="min-h-[44px] min-w-[44px] p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors flex items-center justify-center active:scale-95"
          aria-label="Close Fullscreen Command Center"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Main Grid Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6 my-auto py-4 sm:py-6">
        {/* Left Column: Large Live Camera Feed (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col gap-3">
          <div className="relative aspect-[4/3] sm:aspect-video rounded-2xl bg-black border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center">
            {selectedSource === 'webcam' && videoRef.current ? (
              <video
                src=""
                ref={(el) => {
                  if (el && videoRef.current && videoRef.current.srcObject) {
                    el.srcObject = videoRef.current.srcObject;
                    el.play().catch(() => {});
                  }
                }}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
              />
            ) : (
              <img
                src={activeSampleImage}
                alt="Active Sample Evaluation"
                className="w-full h-full object-contain sm:object-cover"
              />
            )}

            {/* Camera Overlay HUD */}
            <div className="absolute top-3 left-3 right-3 flex justify-between text-xs font-mono text-cyan-300 pointer-events-none">
              <span className="px-2 py-1 rounded bg-black/75 border border-slate-800">
                {selectedSource === 'webcam' ? 'CAM · 640×480 · RAW RGB' : 'BENCHMARK SAMPLE'}
              </span>
              <span className="px-2 py-1 rounded bg-black/75 border border-slate-800 text-emerald-400">
                ACTIVE
              </span>
            </div>

            {/* Bounding box indicator */}
            {hasFace && prediction?.bbox && (
              <div className="absolute inset-x-0 bottom-3 flex justify-center pointer-events-none">
                <span className="px-3 py-1 rounded-full bg-slate-950/85 border border-cyan-500/40 text-cyan-300 font-mono text-xs">
                  FACE ROI: [{prediction.bbox.join(', ')}]
                </span>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs font-mono text-slate-400 px-1 gap-2">
            <span>MODEL: A4 MULTI-REPRESENTATION FUSION</span>
            <span>FPS: {fps.toFixed(1)} · LATENCY: {latencyMs.toFixed(1)} MS</span>
          </div>
        </div>

        {/* Right Column: Prominent Expression, Probabilities & Telemetry (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Dominant Prediction Focal Card */}
          <div className="p-4 sm:p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl relative overflow-hidden">
            <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase block mb-1">
              DOMINANT PREDICTION
            </span>
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
                  <span>{dominantExpr.toUpperCase()}</span>
                  <span>{emoji}</span>
                </h2>
                <p className="text-xs text-slate-400 mt-1 font-mono">
                  RAF-DB 7-CLASS CLASSIFIER
                </p>
              </div>
              <div className="text-right shrink-0">
                <span className="font-mono text-3xl sm:text-4xl font-bold text-cyan-400">
                  {dominantConf}%
                </span>
              </div>
            </div>
          </div>

          {/* 7-Class Probabilities */}
          <div className="p-4 sm:p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2">
            <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase block pb-1 border-b border-slate-800">
              CLASS PROBABILITIES
            </span>

            <div className="space-y-2">
              {CANONICAL_EXPRESSIONS.map((expr) => {
                const prob = hasFace && prediction?.probabilities ? (prediction.probabilities[expr] || 0) : 0;
                const percent = (prob * 100).toFixed(1);
                const isDominant = hasFace && prediction?.prediction === expr;

                return (
                  <div key={expr} className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className={`font-medium ${isDominant ? 'text-cyan-300 font-bold' : 'text-slate-300'}`}>
                        {EXPRESSION_EMOJIS[expr]} {expr}
                      </span>
                      <span className={`font-mono ${isDominant ? 'text-cyan-400 font-bold' : 'text-slate-400'}`}>
                        {percent}%
                      </span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-slate-950 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isDominant ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.7)]' : 'bg-slate-700/60'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, parseFloat(percent)))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Representation Branches Status */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 font-mono text-xs text-slate-300 space-y-1.5">
            <div className="text-cyan-400 font-bold uppercase text-[10px] pb-1 border-b border-slate-800">
              REPRESENTATION PIPELINE STATUS
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Spatial Branch:</span>
              <span className="text-emerald-400">ACTIVE (ConvNeXt-Tiny)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Frequency Branch:</span>
              <span className="text-emerald-400">ACTIVE (2D-FFT Spectral)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Geometry Branch:</span>
              <span className="text-emerald-400">ACTIVE (62-D Blendshapes)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex flex-col sm:flex-row justify-between items-center gap-3 pt-3 sm:pt-4 border-t border-slate-800 text-xs font-mono text-slate-500">
        <span>MULTIVIEW COMMAND CENTER · SPECIFICATION A4</span>
        <button
          onClick={onClose}
          className="w-full sm:w-auto min-h-[44px] px-5 py-2 rounded-lg bg-cyan-500 text-slate-950 font-bold hover:bg-cyan-400 transition-colors active:scale-95"
        >
          DISMISS COMMAND CENTER
        </button>
      </div>
    </div>
  );
};
