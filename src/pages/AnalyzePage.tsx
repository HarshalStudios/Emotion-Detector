import React, { useState, useEffect, useRef, useCallback } from 'react';
import { CameraViewport, CameraStatus, CameraErrorInfo } from '../components/CameraViewport';
import { PredictionPanel } from '../components/PredictionPanel';
import { TelemetryPanel } from '../components/TelemetryPanel';
import { RepresentationWorkspace } from '../components/RepresentationWorkspace';
import { FullscreenMultiviewModal } from '../components/FullscreenMultiviewModal';
import { apiService, PredictResponse } from '../services/api';
import { Play, Square, Maximize2, Activity } from 'lucide-react';

interface AnalyzePageProps {
  onRecordHistory: (record: {
    timestamp: string;
    prediction: string;
    confidence: number;
    latencyMs: number;
    geometryValid: boolean;
  }) => void;
}

export const AnalyzePage: React.FC<AnalyzePageProps> = ({ onRecordHistory }) => {
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [errorInfo, setErrorInfo] = useState<CameraErrorInfo | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [selectedSource, setSelectedSource] = useState<'webcam' | 'sample'>('webcam');
  const [showMeshOverlay, setShowMeshOverlay] = useState<boolean>(true);
  const [isFullscreenModalOpen, setIsFullscreenModalOpen] = useState<boolean>(false);

  // Live Telemetry
  const [framesProcessed, setFramesProcessed] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

  // Check health on mount
  useEffect(() => {
    apiService.checkHealth().then((res) => {
      setBackendConnected(res.connected);
    });
  }, []);

  // Authoritative Prediction State
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  // DOM Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Inference Concurrency Guards
  const isRequestRunningRef = useRef<boolean>(false);
  const loopTimerRef = useRef<number | null>(null);
  const lastLoopTimeRef = useRef<number>(performance.now());
  const loopFrameCounterRef = useRef<number>(0);

  // 1. Robust Camera Acquisition with Explicit Diagnostic Error Categorization
  const handleStartCamera = useCallback(async () => {
    setErrorInfo(null);

    if (selectedSource === 'sample') {
      setCameraStatus('live');
      setIsAnalyzing(true);
      return;
    }

    // Verify browser mediaDevices support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus('error');
      setErrorInfo({
        type: 'security',
        title: 'Camera access unavailable',
        message: 'Camera access requires a secure browser context (HTTPS or localhost) and modern mediaDevices support.',
      });
      return;
    }

    setCameraStatus('requesting');

    try {
      // Use standard specification constraints: facingMode "user", audio false
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play().catch((playErr) => {
          console.warn('Video playback autoplay notice:', playErr);
        });
      }

      setCameraStatus('live');
      setIsAnalyzing(true);
    } catch (err: unknown) {
      console.warn('Camera acquisition error:', err);
      const errorObj = err as { name?: string; message?: string };
      const errName = errorObj.name || '';

      setCameraStatus('error');

      if (errName === 'NotAllowedError' || errName === 'PermissionDeniedError') {
        setErrorInfo({
          type: 'permission_denied',
          title: 'Camera permission required',
          message: 'Allow camera access in your browser\'s site permissions, then try again.',
        });
      } else if (errName === 'NotFoundError' || errName === 'DevicesNotFoundError') {
        setErrorInfo({
          type: 'not_found',
          title: 'No camera detected',
          message: 'No video capture device was found on this system.',
        });
      } else if (errName === 'NotReadableError' || errName === 'TrackStartError') {
        setErrorInfo({
          type: 'in_use',
          title: 'Camera unavailable',
          message: 'Another application may already be using this camera or the hardware is busy.',
        });
      } else if (errName === 'OverconstrainedError') {
        setErrorInfo({
          type: 'overconstrained',
          title: 'Camera constraints unsupported',
          message: 'The requested camera constraints cannot be satisfied by your video device.',
        });
      } else if (errName === 'SecurityError') {
        setErrorInfo({
          type: 'security',
          title: 'Camera access unavailable',
          message: 'Camera access requires a secure browser context.',
        });
      } else {
        setErrorInfo({
          type: 'unknown',
          title: 'Camera unavailable',
          message: errorObj.message || 'Unable to access video stream. Please check camera connection.',
        });
      }
    }
  }, [selectedSource]);

  // 2. Controlled Camera Teardown
  const handleStopCamera = useCallback(() => {
    setIsAnalyzing(false);
    setCameraStatus('idle');

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }
  }, []);

  // 3. Switch between Webcam and Reference Sample Photo
  const handleSelectSource = (source: 'webcam' | 'sample') => {
    if (source === selectedSource) return;
    if (cameraStatus === 'live') {
      handleStopCamera();
    }
    setSelectedSource(source);
    setErrorInfo(null);
  };

  // 4. Teardown only on true unmount
  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    offscreenCanvasRef.current = canvas;

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
      }
    };
  }, []);

  // 5. Inference Loop: Target ~380ms (~2.6-3 inferences/sec) with Concurrency Lock
  useEffect(() => {
    if (!isAnalyzing || cameraStatus !== 'live') {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
      return;
    }

    const intervalMs = 380; // ~2.63 FPS target

    const performInference = async () => {
      // Strictly prevent overlapping requests
      if (isRequestRunningRef.current) {
        return; // Skip if previous request is still in-flight
      }

      const canvas = offscreenCanvasRef.current;
      const video = videoRef.current;
      const sampleImg = document.querySelector('img[src="/test_face.jpg"]') as HTMLImageElement | null;
      const mediaElement = selectedSource === 'webcam' ? video : sampleImg;

      if (!canvas || !mediaElement) return;
      if (selectedSource === 'webcam' && (video?.readyState || 0) < 2) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      isRequestRunningRef.current = true;
      const startTime = performance.now();

      try {
        ctx.drawImage(mediaElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              isRequestRunningRef.current = false;
              return;
            }

            try {
              // POST /predict as multipart/form-data
              const res = await apiService.predictImage(blob);
              const duration = performance.now() - startTime;

              setLatencyMs(duration);
              setBackendConnected(res.backend_connected !== false);
              setPrediction(res);
              setFramesProcessed((prev) => prev + 1);

              if (res.face_detected) {
                onRecordHistory({
                  timestamp: new Date().toLocaleTimeString(),
                  prediction: res.prediction,
                  confidence: res.confidence,
                  latencyMs: duration,
                  geometryValid: res.geometry_valid,
                });
              }

              // Update FPS counter
              loopFrameCounterRef.current++;
              const now = performance.now();
              const elapsed = (now - lastLoopTimeRef.current) / 1000;
              if (elapsed >= 1.0) {
                setFps(loopFrameCounterRef.current / elapsed);
                loopFrameCounterRef.current = 0;
                lastLoopTimeRef.current = now;
              }
            } catch (err) {
              console.warn('Inference request error:', err);
            } finally {
              isRequestRunningRef.current = false;
            }
          },
          'image/jpeg',
          0.85
        );
      } catch (err) {
        console.warn('Frame snapshot error:', err);
        isRequestRunningRef.current = false;
      }
    };

    performInference();
    const timerId = window.setInterval(performInference, intervalMs);
    loopTimerRef.current = timerId;

    return () => {
      clearInterval(timerId);
      loopTimerRef.current = null;
      isRequestRunningRef.current = false;
    };
  }, [isAnalyzing, cameraStatus, selectedSource, onRecordHistory]);

  return (
    <div className="space-y-6 py-2">
      
      {/* Action Header / Toolbar */}
      <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white tracking-wide font-mono uppercase">
              EMOTION DETECTOR WORKSTATION
            </h2>
            <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
              <span>Input:</span>
              <span className="font-mono text-cyan-300 font-semibold">
                {selectedSource === 'webcam' ? 'Live Optical Stream' : 'Reference Portrait Photo'}
              </span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          {cameraStatus === 'live' ? (
            <button
              onClick={handleStopCamera}
              className="px-4 py-2 rounded-lg bg-rose-950/80 border border-rose-500/60 hover:bg-rose-900 text-rose-200 text-xs font-mono font-bold transition-all flex items-center gap-1.5 shadow-sm active:scale-95"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>STOP LIVE ANALYSIS</span>
            </button>
          ) : (
            <button
              onClick={handleStartCamera}
              className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-bold tracking-wide transition-all shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)] flex items-center gap-1.5 active:scale-95"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>START LIVE ANALYSIS</span>
            </button>
          )}

          <button
            onClick={() => setIsFullscreenModalOpen(true)}
            className="px-3.5 py-2 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white text-xs font-mono transition-colors flex items-center gap-1.5"
            title="Launch Fullscreen Command Center"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">FULLSCREEN</span>
          </button>
        </div>
      </section>

      {/* Main Two-Column Workstation: Camera Hero Left (7 Cols), Prediction & Telemetry Right (5 Cols) */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Column: Hero Camera Viewport (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <CameraViewport
            status={cameraStatus}
            errorInfo={errorInfo}
            onStartCamera={handleStartCamera}
            onStopCamera={handleStopCamera}
            prediction={prediction}
            videoRef={videoRef}
            showMeshOverlay={showMeshOverlay}
            onToggleMeshOverlay={() => setShowMeshOverlay(!showMeshOverlay)}
            onOpenFullscreen={() => setIsFullscreenModalOpen(true)}
            selectedSource={selectedSource}
            onSelectSource={handleSelectSource}
          />

          <TelemetryPanel
            prediction={prediction}
            latencyMs={latencyMs}
            fps={fps}
            backendConnected={backendConnected}
          />
        </div>

        {/* Right Column: Prediction & Probabilities (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <PredictionPanel
            prediction={prediction}
            cameraActive={cameraStatus === 'live'}
          />
        </div>

      </section>

      {/* Representation Workspace Section */}
      <section>
        <RepresentationWorkspace
          prediction={prediction}
          cameraActive={cameraStatus === 'live'}
        />
      </section>

      {/* Fullscreen Modal */}
      <FullscreenMultiviewModal
        isOpen={isFullscreenModalOpen}
        onClose={() => setIsFullscreenModalOpen(false)}
        prediction={prediction}
        videoRef={videoRef}
        selectedSource={selectedSource}
        fps={fps}
        latencyMs={latencyMs}
      />

    </div>
  );
};
