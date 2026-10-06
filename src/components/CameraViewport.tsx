import React, { useEffect, useRef } from 'react';
import { Camera, AlertCircle, RefreshCw, Eye, EyeOff, Maximize2, ShieldCheck, Image as ImageIcon, Video, CheckCircle2 } from 'lucide-react';
import { PredictResponse } from '../services/api';

export type CameraErrorType =
  | 'permission_denied'
  | 'not_found'
  | 'in_use'
  | 'overconstrained'
  | 'security'
  | 'unknown';

export interface CameraErrorInfo {
  type: CameraErrorType;
  title: string;
  message: string;
}

export type CameraStatus = 'idle' | 'requesting' | 'live' | 'error';

interface CameraViewportProps {
  status: CameraStatus;
  errorInfo: CameraErrorInfo | null;
  onStartCamera: () => void;
  onStopCamera: () => void;
  prediction: PredictResponse | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  showMeshOverlay: boolean;
  onToggleMeshOverlay: () => void;
  onOpenFullscreen: () => void;
  selectedSource: 'webcam' | 'sample';
  onSelectSource: (source: 'webcam' | 'sample') => void;
  activeSampleImage?: string;
  activeSampleTitle?: string;
}

export const CameraViewport: React.FC<CameraViewportProps> = ({
  status,
  errorInfo,
  onStartCamera,
  onStopCamera,
  prediction,
  videoRef,
  showMeshOverlay,
  onToggleMeshOverlay,
  onOpenFullscreen,
  selectedSource,
  onSelectSource,
  activeSampleImage = '/samples/neutral/neutral_01.jpg',
  activeSampleTitle = 'Sample Portrait',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Render authoritative Bounding Box on canvas overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (status !== 'live' || !showMeshOverlay || !prediction) return;

    const w = canvas.width;
    const h = canvas.height;

    // Use backend authoritative bbox [x, y, width, height]
    if (prediction.face_detected && prediction.bbox) {
      const [bx, by, bw, bh] = prediction.bbox;

      // Scale to canvas coordinate system based on active source dimensions
      const origW = selectedSource === 'sample' ? 512 : (videoRef.current?.videoWidth || 640);
      const origH = selectedSource === 'sample' ? 512 : (videoRef.current?.videoHeight || 480);
      const scaleX = w / origW;
      const scaleY = h / origH;

      const sx = bx * scaleX;
      const sy = by * scaleY;
      const sw = bw * scaleX;
      const sh = bh * scaleY;

      // Clean thin technical bounding box with corner brackets
      ctx.strokeStyle = prediction.partial_face ? '#fbbf24' : '#22d3ee';
      ctx.lineWidth = 2;

      const corner = Math.min(sw, sh) * 0.18;

      // Top-Left corner
      ctx.beginPath();
      ctx.moveTo(sx, sy + corner);
      ctx.lineTo(sx, sy);
      ctx.lineTo(sx + corner, sy);
      // Top-Right corner
      ctx.moveTo(sx + sw - corner, sy);
      ctx.lineTo(sx + sw, sy);
      ctx.lineTo(sx + sw, sy + corner);
      // Bottom-Left corner
      ctx.beginPath();
      ctx.moveTo(sx, sy + sh - corner);
      ctx.lineTo(sx, sy + sh);
      ctx.lineTo(sx + corner, sy + sh);
      // Bottom-Right corner
      ctx.moveTo(sx + sw - corner, sy + sh);
      ctx.lineTo(sx + sw, sy + sh);
      ctx.lineTo(sx + sw, sy + sh - corner);
      ctx.stroke();

      // Delicate center reticle
      const cx = sx + sw / 2;
      const cy = sy + sh / 2;
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
      ctx.beginPath();
      ctx.moveTo(cx - 6, cy);
      ctx.lineTo(cx + 6, cy);
      ctx.moveTo(cx, cy - 6);
      ctx.lineTo(cx, cy + 6);
      ctx.stroke();

      // Technical label stamp
      ctx.fillStyle = 'rgba(2, 6, 23, 0.88)';
      ctx.fillRect(sx, Math.max(0, sy - 18), 110, 16);
      ctx.fillStyle = '#22d3ee';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(
        `ROI · ${(prediction.detection_confidence * 100).toFixed(0)}%`,
        sx + 4,
        Math.max(12, sy - 5)
      );
    }
  }, [status, showMeshOverlay, prediction, selectedSource, videoRef]);

  const hasFace = prediction?.face_detected;
  const isPartial = prediction?.partial_face;
  const isGeometryValid = prediction?.geometry_valid;

  return (
    <div className="relative w-full rounded-2xl bg-slate-900/90 border border-slate-800 p-3 sm:p-4 backdrop-blur-xl shadow-2xl flex flex-col space-y-2.5">
      {/* Top HUD Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-800/80">
        {/* Left: Compact Status Signals */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Live Indicator */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-950 border border-slate-800 font-mono text-[10px]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                status === 'live' ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'
              }`}
            />
            <span className={status === 'live' ? 'text-emerald-300 font-semibold' : 'text-slate-500'}>
              {status === 'live' ? 'ACTIVE' : 'IDLE'}
            </span>
          </div>

          {/* RGB Signal */}
          <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800 font-mono text-[10px] text-cyan-400 font-semibold">
            RGB
          </span>

          {/* Face Signal */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-950 border border-slate-800 font-mono text-[10px]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                status !== 'live'
                  ? 'bg-slate-600'
                  : hasFace
                  ? 'bg-emerald-400'
                  : isPartial
                  ? 'bg-amber-400'
                  : 'bg-slate-600'
              }`}
            />
            <span
              className={
                status !== 'live'
                  ? 'text-slate-500'
                  : hasFace
                  ? 'text-emerald-300'
                  : isPartial
                  ? 'text-amber-300'
                  : 'text-slate-500'
              }
            >
              FACE
            </span>
          </div>

          {/* Geometry Signal */}
          <div className="hidden xs:flex items-center gap-1.5 px-2 py-1 rounded bg-slate-950 border border-slate-800 font-mono text-[10px]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                status === 'live' && isGeometryValid ? 'bg-emerald-400' : 'bg-slate-600'
              }`}
            />
            <span
              className={
                status === 'live' && isGeometryValid ? 'text-emerald-300' : 'text-slate-500'
              }
            >
              GEOM
            </span>
          </div>
        </div>

        {/* Right: Viewport Controls with Touch-Friendly Hitboxes (min-h-[38px]) */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Source switch: Webcam vs Sample Image */}
          <button
            onClick={() => onSelectSource(selectedSource === 'webcam' ? 'sample' : 'webcam')}
            className={`min-h-[38px] px-2.5 py-1.5 rounded-lg border text-xs font-mono transition-colors flex items-center gap-1.5 active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400 ${
              selectedSource === 'webcam'
                ? 'bg-slate-950 border-cyan-500/40 text-cyan-300 hover:border-cyan-400'
                : 'bg-amber-950/40 border-amber-500/40 text-amber-300 hover:border-amber-400'
            }`}
            title="Toggle between live optical webcam and reference test samples"
          >
            {selectedSource === 'webcam' ? (
              <>
                <Video className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <span className="font-semibold">Webcam</span>
              </>
            ) : (
              <>
                <ImageIcon className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <span className="font-semibold">Sample Mode</span>
              </>
            )}
          </button>

          {/* Toggle BBox Mesh Overlay */}
          <button
            onClick={onToggleMeshOverlay}
            className={`min-h-[38px] min-w-[38px] p-2 rounded-lg border text-xs font-mono transition-colors flex items-center justify-center active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400 ${
              showMeshOverlay
                ? 'bg-cyan-950/60 border-cyan-500/50 text-cyan-300'
                : 'bg-slate-950 border-slate-800 text-slate-500 hover:text-slate-300'
            }`}
            title="Toggle Visual Bounding Box"
            aria-label="Toggle Bounding Box"
          >
            {showMeshOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>

          {/* Fullscreen Modal Trigger */}
          <button
            onClick={onOpenFullscreen}
            className="min-h-[38px] min-w-[38px] p-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 hover:text-white transition-colors flex items-center justify-center active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400"
            title="Open Fullscreen Viewport"
            aria-label="Open Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Video/Image Canvas Viewport with fluid aspect ratio */}
      <div className="relative w-full aspect-[4/3] sm:aspect-video rounded-xl bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center">
        {/* Live Video Element */}
        {selectedSource === 'webcam' ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover transform -scale-x-100 transition-opacity duration-300 ${
              status === 'live' ? 'opacity-100' : 'opacity-0'
            }`}
          />
        ) : (
          <div className="relative w-full h-full flex items-center justify-center bg-slate-950">
            <img
              src={activeSampleImage}
              alt={activeSampleTitle}
              crossOrigin="anonymous"
              className="w-full h-full object-contain sm:object-cover transition-opacity duration-200"
            />
            {/* Active Sample Indicator Watermark */}
            <div className="absolute top-3 left-3 bg-slate-950/85 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800 font-mono text-[10px] text-amber-300 flex items-center gap-1.5 shadow-md pointer-events-none">
              <ImageIcon className="w-3 h-3 text-amber-400" />
              <span>TEST SAMPLE: {activeSampleTitle}</span>
            </div>
          </div>
        )}

        {/* BBox Overlay Canvas */}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Idle State Overlay */}
        {status === 'idle' && selectedSource === 'webcam' && (
          <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-4 sm:p-6 text-center space-y-3">
            <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 shadow-inner">
              <Camera className="w-6 h-6 text-cyan-400" />
            </div>
            <div className="max-w-xs space-y-1">
              <h3 className="font-semibold text-slate-100 text-sm tracking-tight">Camera Feed Inactive</h3>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                Enable webcam to stream live RGB frames, or switch to the sample gallery to test photos.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <button
                onClick={onStartCamera}
                className="min-h-[44px] px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold tracking-wide transition-all shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)] active:scale-95 flex items-center gap-2 focus-visible:outline-2 focus-visible:outline-cyan-400"
              >
                <Camera className="w-4 h-4" />
                <span>START CAMERA</span>
              </button>
              <button
                onClick={() => onSelectSource('sample')}
                className="min-h-[44px] px-4 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold tracking-wide transition-colors active:scale-95 flex items-center gap-1.5 focus-visible:outline-2 focus-visible:outline-cyan-400"
              >
                <ImageIcon className="w-4 h-4 text-amber-400" />
                <span>SAMPLE PHOTOS</span>
              </button>
            </div>
          </div>
        )}

        {/* Requesting State Overlay */}
        {status === 'requesting' && (
          <div className="absolute inset-0 bg-slate-950/95 flex flex-col items-center justify-center p-6 text-center space-y-3">
            <RefreshCw className="w-7 h-7 text-cyan-400 animate-spin" />
            <p className="font-mono text-xs text-slate-200 tracking-wider uppercase font-semibold">
              REQUESTING CAMERA ACCESS...
            </p>
            <p className="text-[11px] text-slate-400 max-w-xs">
              Please grant camera permission in your browser prompt to proceed.
            </p>
          </div>
        )}

        {/* Diagnostic Error State Overlays with Clear Fallback Workflow */}
        {status === 'error' && errorInfo && (
          <div className="absolute inset-0 bg-slate-950/95 flex flex-col items-center justify-center p-4 sm:p-6 text-center space-y-3">
            <AlertCircle className="w-8 h-8 text-rose-400" />
            <div className="space-y-1 max-w-sm">
              <h3 className="font-semibold text-rose-200 text-sm">{errorInfo.title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">{errorInfo.message}</p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <button
                onClick={onStartCamera}
                className="min-h-[44px] px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white font-mono active:scale-95"
              >
                TRY AGAIN
              </button>
              <button
                onClick={() => onSelectSource('sample')}
                className="min-h-[44px] px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold font-mono active:scale-95 flex items-center gap-1.5"
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>USE SAMPLE PHOTOS</span>
              </button>
            </div>
          </div>
        )}

        {/* Live Status Tag on Bottom Left */}
        {status === 'live' && (
          <div className="absolute bottom-2.5 left-2.5 flex items-center gap-2 pointer-events-none">
            {!hasFace ? (
              <span className="px-2 py-0.5 rounded bg-slate-950/85 border border-slate-800 font-mono text-[10px] text-slate-400 font-semibold tracking-wider">
                NO FACE DETECTED
              </span>
            ) : isPartial ? (
              <span className="px-2 py-0.5 rounded bg-amber-950/85 border border-amber-600/60 font-mono text-[10px] text-amber-300 font-semibold tracking-wider">
                PARTIAL FACE DETECTED
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded bg-emerald-950/85 border border-emerald-600/60 font-mono text-[10px] text-emerald-300 font-semibold tracking-wider flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                FACE DETECTED
              </span>
            )}
          </div>
        )}
      </div>

      {/* Viewport Privacy & Model Info Notice */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="truncate">Frames analyzed in real-time by the A4 model pipeline. No facial identity stored.</span>
        </div>
        <span className="font-mono text-[10px] text-slate-500 hidden sm:inline">640×480 RGB</span>
      </div>
    </div>
  );
};
