import React from 'react';
import { ArrowRight, Sparkles, Database, Layers, Cpu, Terminal, Activity, CheckCircle2 } from 'lucide-react';
import { PipelineBar } from '../components/PipelineBar';

interface HomePageProps {
  onNavigate: (tab: 'analyze' | 'how-it-works' | 'research') => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-12 py-4">
      {/* 1. Large Editorial Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-slate-900/80 border border-slate-800 p-8 sm:p-12 lg:p-16 backdrop-blur-xl">
        {/* Subtle Ambient Radial Glow */}
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-3xl space-y-6 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-xs font-mono font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            EMOTION DETECTOR // A4 TRI-REPRESENTATION ARCHITECTURE
          </div>

          <div className="space-y-2">
            <h2 className="text-sm font-mono tracking-widest text-cyan-400 uppercase font-semibold">
              EMOTION DETECTOR
            </h2>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-tight">
              One camera.
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-300">
                Three representations.
              </span>
              <br />
              Seven expressions.
            </h1>
          </div>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed font-sans max-w-2xl">
            A real-time facial-expression analysis system combining spatial, frequency and facial-geometry representations through an A4 fusion architecture.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => onNavigate('analyze')}
              className="px-6 py-3.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm tracking-wide transition-all shadow-[0_0_20px_-3px_rgba(6,182,212,0.5)] flex items-center gap-2 active:scale-95"
            >
              <span>Launch Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => onNavigate('how-it-works')}
              className="px-6 py-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 font-semibold text-sm transition-colors flex items-center gap-2"
            >
              <span>How It Works</span>
            </button>
          </div>
        </div>
      </section>

      {/* 2. Minimal Verified Metrics */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
            DATASET
          </span>
          <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-mono">
            RAF-DB
          </div>
          <span className="text-xs text-cyan-400 font-mono mt-1 block">7 CLASS CANONICAL</span>
        </div>

        {/* Metric 2 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
            BRANCHES
          </span>
          <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-mono">
            3
          </div>
          <span className="text-xs text-indigo-300 font-mono mt-1 block">REPRESENTATIONS</span>
        </div>

        {/* Metric 3 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
            CLASSES
          </span>
          <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-mono">
            7
          </div>
          <span className="text-xs text-emerald-400 font-mono mt-1 block">EXPRESSIONS</span>
        </div>

        {/* Metric 4 */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
            DEPLOYMENT
          </span>
          <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-mono">
            ONNX
          </div>
          <span className="text-xs text-amber-300 font-mono mt-1 block">CPU RUNTIME</span>
        </div>
      </section>

      {/* 3. Live-Looking Technical Pipeline Terminal Preview */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Terminal Window (7 Cols) */}
        <div className="lg:col-span-7 rounded-2xl bg-slate-950 border border-slate-800 p-5 shadow-2xl flex flex-col font-mono text-xs">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
              <span className="text-slate-400 text-[11px] ml-2">a4_inference_pipeline.sh</span>
            </div>
            <span className="text-[10px] text-cyan-400 font-semibold">FASTAPI // READY</span>
          </div>

          <div className="space-y-1.5 text-slate-300 leading-relaxed overflow-x-auto">
            <p className="text-cyan-400 font-semibold">&gt; INITIALIZING EMOTION DETECTOR PIPELINE...</p>
            <p>&gt; face_pipeline: MediaPipe 5-Point Roll Alignment (1.30x Margin)</p>
            <p>&gt; tensor_dimensions: [1, 3, 224, 224] float32 RGB (ImageNet norm)</p>
            <p>&gt; branch_01_spatial: ConvNeXt-Tiny Deep Visual Embedding</p>
            <p>&gt; branch_02_frequency: 2D-FFT Spectral Magnitude Transform</p>
            <p>&gt; branch_03_geometry: MediaPipe 62-D Vector (52 Blendshapes + 10 Ratios)</p>
            <p>&gt; fusion_layer: A4 Concatenated Representation Projection</p>
            <p>&gt; provider: ONNXRuntime CPUExecutionProvider (Thread-Optimized)</p>
            <div className="my-2 p-2.5 rounded bg-slate-900 border border-slate-800 text-slate-200">
              <p className="text-emerald-400 font-bold">&gt; face_detected: true</p>
              <p className="text-emerald-400">&gt; geometry_valid: true</p>
              <p className="text-cyan-300 font-bold">&gt; candidate: A4_SPATIAL_FREQUENCY_GEOMETRY</p>
              <p className="text-white font-extrabold">&gt; predicted_expression: HAPPY</p>
              <p className="text-cyan-400 font-bold">&gt; confidence: 82.4%</p>
            </div>
            <p className="text-slate-500 text-[10px] pt-1">
              [Note: Terminal values illustrate active A4 model pipeline architecture]
            </p>
          </div>
        </div>

        {/* System Capabilities (5 Cols) */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 backdrop-blur-xl flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono font-semibold uppercase mb-1">
              <Activity className="w-4 h-4" />
              TRI-REPRESENTATION FUSION
            </div>
            <h3 className="text-xl font-bold text-white tracking-tight">
              Complementary Vision Channels
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed font-sans mt-2">
              Human facial expressions generate both macro-morphological shifts (geometric blendshapes) and subtle micro-texture tension variations (frequency spectral response). By fusing spatial visual features with Fourier spectra and facial geometry, the A4 model achieves superior stability across diverse camera conditions.
            </p>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-slate-400">Spatial Convolution:</span>
              <span className="text-white font-semibold">ConvNeXt-Tiny</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-slate-400">Frequency Transform:</span>
              <span className="text-white font-semibold">2D Real FFT</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800">
              <span className="text-slate-400">Facial Geometry:</span>
              <span className="text-emerald-400 font-bold">62-D Vector</span>
            </div>
          </div>

          <button
            onClick={() => onNavigate('analyze')}
            className="w-full py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold font-mono tracking-wider transition-all shadow-sm"
          >
            OPEN LIVE WORKSTATION →
          </button>
        </div>

      </section>

      {/* 4. Technical Pipeline Bar */}
      <PipelineBar currentStep={6} />
    </div>
  );
};
