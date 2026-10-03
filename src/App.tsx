/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Cpu, CheckCircle2, ShieldCheck, Activity, BarChart3, Database, Layers } from 'lucide-react';

export default function App() {
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
      status: "Verified",
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
      status: "Verified",
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
      status: "Verified",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600/20 p-2 rounded-lg border border-indigo-500/30 text-indigo-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">Emotion Detector Research Lab</h1>
              <p className="text-xs text-slate-400">Multi-Representation Facial Mood Analysis from a Single RGB Camera</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
              Step 3 Completed
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              Gate 2 Benchmarked
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Research Context Card */}
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-white mb-2">Research Directive & Scope</h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            Investigating whether combining complementary facial representations (Spatial CNN + Frequency 2D-FFT + MediaPipe 62-D Geometry with learned gated fusion) improves accuracy, out-of-distribution robustness, and temporal stability over a single RGB camera stream.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-4">
              <div className="flex items-center text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-1">
                <Database className="w-4 h-4 mr-1.5" /> Primary Dataset
              </div>
              <p className="text-sm font-medium text-white">RAF-DB (Basic 7-Class)</p>
              <p className="text-xs text-slate-400 mt-1">9,817 train / 2,454 stratified val (Seed 42); 3,068 test untouched</p>
            </div>
            <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-4">
              <div className="flex items-center text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
                <ShieldCheck className="w-4 h-4 mr-1.5" /> Webcam Evaluation
              </div>
              <p className="text-sm font-medium text-white">10 Consenting Participants</p>
              <p className="text-xs text-slate-400 mt-1">P01–P06 Validation (756 points) | P07–P10 Final Test (504 points)</p>
            </div>
            <div className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-4">
              <div className="flex items-center text-amber-400 text-xs font-semibold uppercase tracking-wider mb-1">
                <Layers className="w-4 h-4 mr-1.5" /> Representations
              </div>
              <p className="text-sm font-medium text-white">Spatial + 2D FFT + 62-D Geometry</p>
              <p className="text-xs text-slate-400 mt-1">52 Blendshapes + 10 normalized landmark ratios</p>
            </div>
          </div>
        </section>

        {/* Step 3A & 3B: Preprocessing Verification */}
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 bg-emerald-500/10 text-emerald-400 rounded-md">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <h2 className="text-lg font-semibold text-white">Step 3B: Preprocessing Verification & Parity Audit</h2>
            </div>
            <span className="text-xs font-mono bg-emerald-950 text-emerald-300 border border-emerald-800/50 px-2 py-0.5 rounded">
              Parity Delta: 0.00000000e+00
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-slate-800/40 p-3 rounded border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Output Tensor</span>
              <span className="font-mono font-semibold text-white">(3, 224, 224) float32</span>
            </div>
            <div className="bg-slate-800/40 p-3 rounded border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Geometry Vector</span>
              <span className="font-mono font-semibold text-white">(62,) float32</span>
            </div>
            <div className="bg-slate-800/40 p-3 rounded border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Alignment Method</span>
              <span className="font-semibold text-white">5-Point Roll-Based Rigid Rotation</span>
            </div>
            <div className="bg-slate-800/40 p-3 rounded border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Margin Factor</span>
              <span className="font-semibold text-white">1.30x Center Scaled</span>
            </div>
          </div>

          <div className="mt-4 p-3 bg-slate-950 border border-slate-800 rounded text-xs font-mono text-slate-300">
            <span className="text-emerald-400 font-bold">[VERIFIED]</span> All 6 automated test suites passed: No-Face Handled, 224x224 Normalized Bounds, 62-D Vector Computed, Training Samples 100% Retained, Webcam Sub-64px Masked, Deterministic Identity Passed.
          </div>
        </section>

        {/* Step 3C: Gate 2 CPU Backbone Benchmark */}
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="p-1.5 bg-indigo-500/10 text-indigo-400 rounded-md">
                <BarChart3 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Step 3C: Gate 2 CPU Backbone Benchmark</h2>
                <p className="text-xs text-slate-400">Standardized ONNX Runtime CPUExecutionProvider (Batch = 1, 1,000 Timed Runs)</p>
              </div>
            </div>
            <div className="text-right text-xs text-slate-400">
              <span className="block font-medium text-slate-300">Target CPU: AMD EPYC 7B12</span>
              <span>8 Cores | 16 GB RAM | Linux 6.6</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                  <th className="py-2.5 px-3">Candidate Backbone</th>
                  <th className="py-2.5 px-3">Params</th>
                  <th className="py-2.5 px-3">ONNX Size</th>
                  <th className="py-2.5 px-3">1-Thread P50</th>
                  <th className="py-2.5 px-3">1-Thread FPS</th>
                  <th className="py-2.5 px-3">4-Thread P50</th>
                  <th className="py-2.5 px-3">4-Thread FPS</th>
                  <th className="py-2.5 px-3">1-Thread P95</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {benchmarkResults.map((b) => (
                  <tr key={b.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3 px-3 font-sans font-medium text-white">{b.name}</td>
                    <td className="py-3 px-3 text-slate-300">{b.params}</td>
                    <td className="py-3 px-3 text-slate-300">{b.size}</td>
                    <td className="py-3 px-3 text-amber-300">{b.p50_1th}</td>
                    <td className="py-3 px-3 text-emerald-400 font-bold">{b.fps_1th}</td>
                    <td className="py-3 px-3 text-cyan-300">{b.p50_4th}</td>
                    <td className="py-3 px-3 text-cyan-400 font-bold">{b.fps_4th}</td>
                    <td className="py-3 px-3 text-slate-400">{b.p95}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 bg-indigo-950/30 border border-indigo-900/40 rounded-lg text-xs text-indigo-200">
            <strong>Decision Rule for Gate 2:</strong> All three candidate backbones comfortably exceed the target threshold of 15 FPS on a single CPU thread (31.4 to 154.3 FPS). Final backbone selection will be paired with validation macro-F1 during baseline training; if candidates are within one standard deviation, the faster candidate wins automatically.
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/40 px-6 py-4 text-center text-xs text-slate-500">
        Multi-Representation Facial Mood Analysis from a Single RGB Camera &bull; Academic Research Project &bull; Dev Server Online (Port 3000)
      </footer>
    </div>
  );
}
