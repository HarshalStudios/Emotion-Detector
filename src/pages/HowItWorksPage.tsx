import React from 'react';
import { Camera, Scan, Crop, Layers, GitMerge, CheckCircle, Cpu, ShieldCheck } from 'lucide-react';
import { PipelineBar } from '../components/PipelineBar';

export const HowItWorksPage: React.FC = () => {
  const pipelineStages = [
    {
      step: '01',
      title: 'Single RGB Optical Acquisition',
      desc: 'Streams standard video frames (640×480) from a standard commodity web camera. The browser processes video locally without storing biometric identity.',
      specs: ['Input: Standard RGB Frame', 'Latency: < 16 ms preview', 'Framerate: 60 FPS independent buffer'],
      icon: Camera,
    },
    {
      step: '02',
      title: 'MediaPipe Face Mesh Localization',
      desc: 'Detects facial presence and extracts 468 3D landmark coordinates to establish spatial bounding boxes and reliable facial anchor points.',
      specs: ['Anchor Points: 5-Point Core Anchors', 'Robustness: Real-time pose tracking', 'Validation: Binary face detection flag'],
      icon: Scan,
    },
    {
      step: '03',
      title: 'Roll Alignment & Margin Crop',
      desc: 'Calculates the inter-ocular rotation angle (roll) and applies a rigid 2D Euclidean rotation to align eyes horizontally, followed by a 1.30× center-scaled margin crop resized to exactly 224×224.',
      specs: ['Resolution: 224×224 px', 'Channels: RGB Float32', 'Parity Delta: 0.00000000e+00 verified'],
      icon: Crop,
    },
    {
      step: '04',
      title: 'Tri-Representation Extraction',
      desc: 'The aligned facial patch is split into three orthogonal representation branches: Spatial (ConvNeXt-Tiny visual tokens), Frequency (2D-FFT spectral magnitude map), and Geometry (62-D blendshape vector).',
      specs: ['Spatial: ConvNeXt Hierarchical Features', 'Frequency: 2D FFT Magnitude Spectrum', 'Geometry: 52 Blendshapes + 10 Ratios'],
      icon: Layers,
    },
    {
      step: '05',
      title: 'Multi-Representation Feature Fusion',
      desc: 'The spatial feature vector, frequency spectral embeddings, and normalized 62-D geometry vector are concatenated and projected through a multi-layer perceptron with dropout regularizers.',
      specs: ['Candidate: A4 Fusion Layer', 'Regularization: Batch Normalization + Dropout', 'Target: 7 Canonical Latent Logits'],
      icon: GitMerge,
    },
    {
      step: '06',
      title: 'ONNX Runtime Inference & 7-Class Output',
      desc: 'The fused graph is executed via ONNX Runtime using CPUExecutionProvider with multi-thread optimizations, yielding normalized Softmax posteriors across seven canonical emotional states.',
      specs: ['Latency: ~13.9 ms (4-Thread AMD EPYC)', 'Throughput: > 70 FPS', 'Posteriors: 7 Canonical Classes'],
      icon: CheckCircle,
    },
  ];

  return (
    <div className="space-y-6 sm:space-y-10 py-4 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="space-y-2.5">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-xs font-mono font-medium">
          SYSTEM ARCHITECTURE SPECIFICATION
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
          How Multi-Representation Expression Analysis Works
        </h1>
        <p className="text-xs sm:text-base text-slate-300 leading-relaxed font-sans max-w-3xl">
          Traditional facial expression classifiers rely purely on static RGB pixels, which are highly sensitive to head pose variations, shadows, and identity traits. The A4 architecture combines spatial convolutions with frequency-domain spectral analysis and 62-dimensional geometric blendshapes.
        </p>
      </div>

      {/* Pipeline Visual Bar */}
      <PipelineBar currentStep={6} />

      {/* Pipeline Steps Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
        {pipelineStages.map((stage) => {
          const Icon = stage.icon;
          return (
            <div
              key={stage.step}
              className="p-5 sm:p-6 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl flex flex-col justify-between space-y-4 shadow-lg hover:border-slate-700 transition-colors"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-cyan-400">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-md border border-cyan-500/30">
                    STAGE {stage.step}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white tracking-tight">{stage.title}</h3>
                <p className="text-xs text-slate-300 leading-relaxed mt-2 font-sans">{stage.desc}</p>
              </div>

              <div className="pt-3 border-t border-slate-800/80 space-y-1">
                {stage.specs.map((spec, i) => (
                  <div key={i} className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                    <span className="w-1 h-1 rounded-full bg-cyan-400" />
                    <span>{spec}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Technical Summary Banner */}
      <div className="p-4 sm:p-6 rounded-2xl bg-slate-900/50 border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        <div className="flex items-center gap-3">
          <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0" />
          <div className="text-xs text-slate-300 leading-relaxed">
            <strong>Zero Biometric Storage:</strong> Camera frames are converted to temporary normalized tensors in memory during active inference. No biometric face vectors or identifiers are permanently retained.
          </div>
        </div>
      </div>
    </div>
  );
};
