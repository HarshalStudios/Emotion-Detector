import React, { useState } from 'react';
import { PredictResponse } from '../services/api';
import { Activity, ChevronDown, ChevronUp, Cpu, Gauge } from 'lucide-react';

interface TelemetryPanelProps {
  prediction: PredictResponse | null;
  latencyMs: number;
  fps: number;
  backendConnected: boolean;
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  prediction,
  latencyMs,
  fps,
  backendConnected,
}) => {
  const [mobileExpanded, setMobileExpanded] = useState<boolean>(false);

  const hasFace = prediction?.face_detected;
  const pitch = hasFace && prediction.head_pose ? prediction.head_pose.pitch : 0;
  const yaw = hasFace && prediction.head_pose ? prediction.head_pose.yaw : 0;
  const roll = hasFace && prediction.head_pose ? prediction.head_pose.roll : 0;
  const conf = hasFace ? (prediction.detection_confidence * 100).toFixed(1) : '0.0';

  const formatAngle = (val: number) => {
    return val > 0 ? `+${val.toFixed(1)}°` : `${val.toFixed(1)}°`;
  };

  return (
    <div className="p-3.5 sm:p-4 rounded-xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl shadow-lg w-full">
      <div className="flex items-center justify-between pb-2 mb-2 sm:mb-3 border-b border-slate-800">
        <span className="font-mono text-xs text-cyan-400 font-semibold tracking-wider uppercase flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5" />
          TECHNICAL TELEMETRY
        </span>

        {/* Mobile Collapse Toggle Button (min 44px tap target) */}
        <button
          onClick={() => setMobileExpanded(!mobileExpanded)}
          className="sm:hidden min-h-[36px] px-2 py-1 rounded bg-slate-950 border border-slate-800 text-[10px] font-mono text-slate-300 flex items-center gap-1 active:scale-95"
          aria-expanded={mobileExpanded}
        >
          <span>{mobileExpanded ? 'HIDE DETAILS' : 'SHOW DETAILS'}</span>
          {mobileExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>

        <span className="hidden sm:inline font-mono text-[10px] text-slate-400">
          REAL-TIME PIPELINE INVARIANTS
        </span>
      </div>

      {/* High-Level Compact Telemetry Ribbon (Always Visible on Mobile and Desktop) */}
      <div className="grid grid-cols-3 sm:hidden gap-2 font-mono text-xs mb-2">
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-slate-400 text-[9px] block uppercase">LATENCY</span>
          <span className="text-amber-300 font-bold">{latencyMs.toFixed(0)} ms</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-slate-400 text-[9px] block uppercase">THROUGHPUT</span>
          <span className="text-emerald-400 font-bold">{fps.toFixed(1)} FPS</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-slate-400 text-[9px] block uppercase">ENGINE</span>
          <span className={backendConnected ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
            {backendConnected ? 'READY' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Detailed Technical Grid (Always Visible on Desktop, Expandable on Mobile) */}
      <div className={`grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-2.5 font-mono text-xs ${
        mobileExpanded ? 'grid' : 'hidden sm:grid'
      }`}>
        {/* Detection Status */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">FACE DETECTION</span>
          <span className={hasFace ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
            {hasFace ? 'DETECTED' : 'SEARCHING'}
          </span>
        </div>

        {/* Geometry Valid */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">GEOMETRY (62-D)</span>
          <span className={prediction?.geometry_valid ? 'text-cyan-400 font-bold' : 'text-slate-500'}>
            {prediction?.geometry_valid ? 'VALID (62-D)' : 'INVALID'}
          </span>
        </div>

        {/* Confidence */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">DETECTION CONF</span>
          <span className="text-white font-semibold">{conf}%</span>
        </div>

        {/* Head Pose */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">HEAD POSE (P/Y/R)</span>
          <span className="text-slate-200 truncate block">
            {formatAngle(pitch)} / {formatAngle(yaw)} / {formatAngle(roll)}
          </span>
        </div>

        {/* Latency */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">INFERENCE LATENCY</span>
          <span className="text-amber-300 font-semibold">{latencyMs.toFixed(1)} ms</span>
        </div>

        {/* FPS */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">FRONTEND LOOP</span>
          <span className="text-emerald-400 font-bold">{fps.toFixed(1)} FPS</span>
        </div>

        {/* Model Architecture */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">CANDIDATE MODEL</span>
          <span className="text-cyan-400 font-semibold">A4 // ONNX FP32</span>
        </div>

        {/* Engine Status */}
        <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80">
          <span className="text-slate-400 text-[10px] block uppercase">BACKEND PIPELINE</span>
          <span className={backendConnected ? 'text-emerald-400 font-bold' : 'text-red-400 font-medium'}>
            {backendConnected ? 'FASTAPI // A4 ONNX' : 'OFFLINE // STANDBY'}
          </span>
        </div>
      </div>
    </div>
  );
};
