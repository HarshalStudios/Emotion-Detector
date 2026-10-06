import React, { useState, useEffect, useRef, useCallback } from 'react';
import { CameraViewport, CameraStatus, CameraErrorInfo } from '../components/CameraViewport';
import { PredictionPanel } from '../components/PredictionPanel';
import { TelemetryPanel } from '../components/TelemetryPanel';
import { RepresentationWorkspace } from '../components/RepresentationWorkspace';
import { FullscreenMultiviewModal } from '../components/FullscreenMultiviewModal';
import { SampleGallery } from '../components/SampleGallery';
import { SAMPLE_IMAGES, SampleImageItem } from '../data/sampleImages';
import { apiService, PredictResponse } from '../services/api';
import { Play, Square, Maximize2, Activity, ImageIcon, Video } from 'lucide-react';

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
  const [selectedSample, setSelectedSample] = useState<SampleImageItem>(SAMPLE_IMAGES[0]);
  const [isEvaluatingSample, setIsEvaluatingSample] = useState<boolean>(false);
  const [showMeshOverlay, setShowMeshOverlay] = useState<boolean>(true);
  const [isFullscreenModalOpen, setIsFullscreenModalOpen] = useState<boolean>(false);

  // Live Telemetry
  const [framesProcessed, setFramesProcessed] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

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

  // Check health on mount
  useEffect(() => {
    apiService.checkHealth().then((res) => {
      setBackendConnected(res.connected);
    });
  }, []);

  // 1. Evaluate a specific sample image through the authoritative backend pipeline
  const evaluateSampleImage = useCallback(async (sample: SampleImageItem) => {
    setIsEvaluatingSample(true);
    const startTime = performance.now();

    try {
      const resp = await fetch(sample.image);
      if (!resp.ok) {
        throw new Error(`Failed to load sample image: ${resp.status}`);
      }
      const blob = await resp.blob();

      // Send to authoritative /predict endpoint
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
    } catch (err) {
      console.warn('Sample evaluation failed:', err);
    } finally {
      setIsEvaluatingSample(false);
    }
  }, [onRecordHistory]);

  // 2. Select Sample from Gallery
  const handleSelectSample = useCallback((sample: SampleImageItem) => {
    // If camera was live, stop it
    if (selectedSource === 'webcam' && streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }

    setSelectedSource('sample');
    setSelectedSample(sample);
    setCameraStatus('live');
    setIsAnalyzing(true);
    setErrorInfo(null);

    // Evaluate immediately with real backend
    evaluateSampleImage(sample);
  }, [selectedSource, evaluateSampleImage]);

  // 3. Robust Camera Acquisition with Diagnostic Error Categorization
  const handleStartCamera = useCallback(async () => {
    setErrorInfo(null);

    if (selectedSource === 'sample') {
      setSelectedSource('webcam');
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
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
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

      setSelectedSource('webcam');
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
          title: 'Camera permission denied',
          message: 'Allow camera access in your browser\'s site permissions, or test any of the 20 benchmark sample photos below.',
        });
      } else if (errName === 'NotFoundError' || errName === 'DevicesNotFoundError') {
        setErrorInfo({
          type: 'not_found',
          title: 'No camera hardware found',
          message: 'No video capture device was detected. You can test the model using reference portrait samples below.',
        });
      } else if (errName === 'NotReadableError' || errName === 'TrackStartError') {
        setErrorInfo({
          type: 'in_use',
          title: 'Camera in use by another app',
          message: 'Another application may be using this camera. Please close other camera apps or switch to sample photos.',
        });
      } else {
        setErrorInfo({
          type: 'unknown',
          title: 'Camera unavailable',
          message: errorObj.message || 'Unable to access video stream. Try using reference portrait samples below.',
        });
      }
    }
  }, [selectedSource]);

  // 4. Controlled Camera Teardown
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

  // 5. Switch between Webcam and Sample Photo
  const handleSelectSource = (source: 'webcam' | 'sample') => {
    if (source === selectedSource && cameraStatus === 'live') return;
    if (cameraStatus === 'live' && selectedSource === 'webcam') {
      handleStopCamera();
    }
    setSelectedSource(source);
    setErrorInfo(null);
    if (source === 'sample') {
      setCameraStatus('live');
      setIsAnalyzing(true);
      evaluateSampleImage(selectedSample);
    }
  };

  // 6. Setup Offscreen Canvas
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

  // 7. Live Webcam Inference Loop (runs only when webcam is active)
  useEffect(() => {
    if (!isAnalyzing || cameraStatus !== 'live' || selectedSource !== 'webcam') {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
      return;
    }

    const intervalMs = 380; // ~2.63 FPS target

    const performInference = async () => {
      if (isRequestRunningRef.current) return;

      const canvas = offscreenCanvasRef.current;
      const video = videoRef.current;

      if (!canvas || !video || (video.readyState || 0) < 2) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      isRequestRunningRef.current = true;
      const startTime = performance.now();

      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              isRequestRunningRef.current = false;
              return;
            }

            try {
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

              loopFrameCounterRef.current++;
              const now = performance.now();
              const elapsed = (now - lastLoopTimeRef.current) / 1000;
              if (elapsed >= 1.0) {
                setFps(loopFrameCounterRef.current / elapsed);
                loopFrameCounterRef.current = 0;
                lastLoopTimeRef.current = now;
              }
            } catch (err) {
              console.warn('Live inference loop error:', err);
            } finally {
              isRequestRunningRef.current = false;
            }
          },
          'image/jpeg',
          0.85
        );
      } catch (err) {
        console.warn('Frame capture error:', err);
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
    <div className="space-y-5 sm:space-y-6 py-2 w-full">
      {/* 1. Action Header / Toolbar */}
      <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 sm:p-4 rounded-xl bg-slate-900/90 border border-slate-800 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-400 shrink-0">
            <Activity className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs sm:text-sm font-bold text-white tracking-wide font-mono uppercase truncate">
              EMOTION DETECTOR WORKSTATION
            </h2>
            <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
              <span>Input Mode:</span>
              <span className="font-mono text-cyan-300 font-semibold truncate">
                {selectedSource === 'webcam' ? 'Live Optical Stream' : `Sample: ${selectedSample.title} (${selectedSample.expectedEmotion})`}
              </span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {cameraStatus === 'live' && selectedSource === 'webcam' ? (
            <button
              onClick={handleStopCamera}
              className="min-h-[40px] px-3.5 sm:px-4 py-2 rounded-lg bg-rose-950/80 border border-rose-500/60 hover:bg-rose-900 text-rose-200 text-xs font-mono font-bold transition-all flex items-center gap-1.5 shadow-sm active:scale-95"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>STOP LIVE STREAM</span>
            </button>
          ) : (
            <button
              onClick={handleStartCamera}
              className="min-h-[40px] px-3.5 sm:px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-mono font-bold tracking-wide transition-all shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)] flex items-center gap-1.5 active:scale-95"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>START CAMERA</span>
            </button>
          )}

          <button
            onClick={() => setIsFullscreenModalOpen(true)}
            className="min-h-[40px] px-3 sm:px-3.5 py-2 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white text-xs font-mono transition-colors flex items-center gap-1.5 active:scale-95"
            title="Launch Fullscreen Viewport"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">FULLSCREEN</span>
          </button>
        </div>
      </section>

      {/* 2. Main Workstation Layout:
          Desktop: 2 Columns (7 cols left for Camera + Telemetry, 5 cols right for Prediction)
          Mobile: Single column where Camera is followed immediately by Prediction Panel */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6 items-start">
        {/* Left Column: Camera Viewport */}
        <div className="lg:col-span-7 flex flex-col gap-4 w-full">
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
            activeSampleImage={selectedSample.image}
            activeSampleTitle={selectedSample.title}
          />

          {/* Desktop Telemetry placement */}
          <div className="hidden lg:block">
            <TelemetryPanel
              prediction={prediction}
              latencyMs={latencyMs}
              fps={fps}
              backendConnected={backendConnected}
            />
          </div>
        </div>

        {/* Right Column: Prediction Panel (Appears immediately below camera on mobile screens) */}
        <div className="lg:col-span-5 flex flex-col gap-4 w-full">
          <PredictionPanel
            prediction={prediction}
            cameraActive={cameraStatus === 'live'}
          />

          {/* Mobile Telemetry placement (collapsible, below prediction so prediction isn't buried) */}
          <div className="block lg:hidden">
            <TelemetryPanel
              prediction={prediction}
              latencyMs={latencyMs}
              fps={fps}
              backendConnected={backendConnected}
            />
          </div>
        </div>
      </section>

      {/* 3. Sample Image Gallery Section (20 Genuine Labeled Benchmark Samples) */}
      <section>
        <SampleGallery
          selectedSampleId={selectedSample.id}
          onSelectSample={handleSelectSample}
          prediction={prediction}
          isEvaluating={isEvaluatingSample}
        />
      </section>

      {/* 4. Tri-Representation Workspace Section (Spatial + Frequency + Geometry) */}
      <section>
        <RepresentationWorkspace
          prediction={prediction}
          cameraActive={cameraStatus === 'live'}
        />
      </section>

      {/* 5. Fullscreen Workstation Modal */}
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
