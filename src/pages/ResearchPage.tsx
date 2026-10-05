import React from 'react';
import { Database, BarChart3, CheckCircle2, Layers, Cpu, ShieldCheck, FileText } from 'lucide-react';

export const ResearchPage: React.FC = () => {
  const benchmarkResults = [
    {
      name: "ConvNeXt-Tiny",
      id: "convnext_tiny",
      params: "28.60 M",
      size: "109.3 MB",
      p50_1th: "31.84 ms",
      fps_1th: "31.4 FPS",
      p50_4th: "13.91 ms",
      fps_4th: "71.9 FPS",
      p95: "34.62 ms",
      status: "Selected Candidate",
    },
    {
      name: "EfficientNet-B0",
      id: "efficientnet_b0",
      params: "5.29 M",
      size: "20.35 MB",
      p50_1th: "10.42 ms",
      fps_1th: "96.0 FPS",
      p50_4th: "5.23 ms",
      fps_4th: "191.2 FPS",
      p95: "11.85 ms",
      status: "Benchmarked",
    },
    {
      name: "MobileNetV3-Large",
      id: "mobilenetv3_large_100",
      params: "4.21 M",
      size: "16.32 MB",
      p50_1th: "6.48 ms",
      fps_1th: "154.3 FPS",
      p50_4th: "3.65 ms",
      fps_4th: "274.0 FPS",
      p95: "7.92 ms",
      status: "Benchmarked",
    },
  ];

  const datasetSplits = [
    { split: 'Train Set', samples: '9,817', percent: '64.0%', desc: 'Stratified basic 7-class samples (Seed 42 deterministic)' },
    { split: 'Validation Set', samples: '2,454', percent: '16.0%', desc: 'Stratified model checkpoint evaluation' },
    { split: 'Test Set', samples: '3,068', percent: '20.0%', desc: 'Official untouched holdout benchmark partition' },
  ];

  return (
    <div className="space-y-8 py-4 max-w-5xl mx-auto">
      {/* Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-xs font-mono font-medium">
          RESEARCH ARCHIVE & BENCHMARK AUDIT
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Experimental Candidates & Dataset Specification
        </h1>
        <p className="text-sm sm:text-base text-slate-300 leading-relaxed font-sans max-w-3xl">
          Complete technical parameters for Candidate A4 (Spatial + Frequency + Geometry Fusion), trained and calibrated on the authoritative RAF-DB 7-class dataset.
        </p>
      </div>

      {/* Dataset Breakdown Card */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <Database className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">Primary Dataset: RAF-DB (Real-world Affective Faces)</h2>
          </div>
          <span className="font-mono text-xs px-2.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30">
            TOTAL: 15,339 SAMPLES
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed font-sans">
          The Real-world Affective Face Database (RAF-DB) contains highly diverse real-world images exhibiting varied lighting, head poses, ethnicities, and occlusions. The basic 7-class partition maps to canonical emotion classes: <em>Neutral, Happy, Sad, Surprise, Fear, Disgust, and Angry</em>.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
          {datasetSplits.map((item) => (
            <div key={item.split} className="p-4 rounded-xl bg-slate-950 border border-slate-800">
              <div className="flex justify-between items-center text-xs font-mono text-cyan-400 mb-1">
                <span className="font-bold">{item.split}</span>
                <span>{item.percent}</span>
              </div>
              <div className="text-2xl font-bold font-mono text-white">{item.samples}</div>
              <p className="text-[11px] text-slate-400 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Step 3B Preprocessing Parity Audit Card */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">Step 3B: Preprocessing Parity Audit</h2>
          </div>
          <span className="font-mono text-xs text-emerald-300 bg-emerald-950 px-2.5 py-0.5 rounded border border-emerald-800/60">
            PARITY DELTA: 0.00000000e+00
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400 text-[10px] block uppercase">OUTPUT TENSOR</span>
            <span className="text-white font-semibold">(3, 224, 224) fp32</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400 text-[10px] block uppercase">GEOMETRY VECTOR</span>
            <span className="text-white font-semibold">(62,) fp32</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400 text-[10px] block uppercase">ALIGNMENT METHOD</span>
            <span className="text-white font-semibold">5-Point Roll Rotation</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400 text-[10px] block uppercase">MARGIN FACTOR</span>
            <span className="text-white font-semibold">1.30x Center Scaled</span>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-xs font-mono text-slate-300">
          <span className="text-emerald-400 font-bold">[VERIFIED]</span> All 6 automated test suites passed: No-Face Handled, 224×224 Normalized Bounds, 62-D Vector Computed, Training Samples 100% Retained, Webcam Sub-64px Masked, Deterministic Identity Passed.
        </div>
      </div>

      {/* Gate 2 CPU Backbone Benchmark Table */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">Gate 2 CPU Backbone Benchmark</h2>
              <p className="text-xs text-slate-400">ONNX Runtime CPUExecutionProvider (Batch = 1, 1,000 Timed Runs)</p>
            </div>
          </div>
          <div className="text-right text-xs font-mono text-slate-400 hidden sm:block">
            <span>Target CPU: AMD EPYC 7B12</span>
            <span className="block text-[10px]">8 Cores · 16 GB RAM</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase font-semibold">
                <th className="py-2.5 px-3">Candidate Backbone</th>
                <th className="py-2.5 px-3">Params</th>
                <th className="py-2.5 px-3">ONNX Size</th>
                <th className="py-2.5 px-3">1-Thread P50</th>
                <th className="py-2.5 px-3">1-Thread FPS</th>
                <th className="py-2.5 px-3">4-Thread P50</th>
                <th className="py-2.5 px-3">4-Thread FPS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-xs">
              {benchmarkResults.map((b) => (
                <tr key={b.id} className="hover:bg-slate-800/30">
                  <td className="py-3 px-3 font-semibold text-white font-sans">{b.name}</td>
                  <td className="py-3 px-3 text-slate-300">{b.params}</td>
                  <td className="py-3 px-3 text-slate-300">{b.size}</td>
                  <td className="py-3 px-3 text-amber-300">{b.p50_1th}</td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">{b.fps_1th}</td>
                  <td className="py-3 px-3 text-cyan-300">{b.p50_4th}</td>
                  <td className="py-3 px-3 text-cyan-400 font-bold">{b.fps_4th}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-900/40 text-xs text-cyan-200">
          <strong>Decision Threshold:</strong> All candidate backbones comfortably exceed the real-time target of 15 FPS on a single CPU thread (31.4 to 154.3 FPS). ConvNeXt-Tiny was adopted for the spatial branch to provide superior semantic visual representations for candidate A4.
        </div>
      </div>
    </div>
  );
};
