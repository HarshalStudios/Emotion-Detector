import React, { useState } from 'react';
import { SAMPLE_IMAGES, SampleImageItem } from '../data/sampleImages';
import { ExpressionType, PredictResponse, EXPRESSION_EMOJIS } from '../services/api';
import { CheckCircle2, AlertTriangle, Play, RefreshCw, Layers } from 'lucide-react';

export interface EvaluatedSampleRecord {
  expected: string;
  predicted: string;
  confidence: number;
  match: boolean;
}

interface SampleGalleryProps {
  selectedSampleId: string;
  onSelectSample: (sample: SampleImageItem) => void;
  prediction: PredictResponse | null;
  isEvaluating: boolean;
  evaluatedSamples: Record<string, EvaluatedSampleRecord>;
  onRunAllBenchmarks?: () => void;
  isBatchEvaluating?: boolean;
  batchProgress?: { current: number; total: number } | null;
}

const CATEGORIES: { label: string; value: 'ALL' | ExpressionType }[] = [
  { label: 'ALL (20)', value: 'ALL' },
  { label: 'NEUTRAL (3)', value: 'Neutral' },
  { label: 'HAPPY (3)', value: 'Happy' },
  { label: 'SAD (3)', value: 'Sad' },
  { label: 'SURPRISE (3)', value: 'Surprise' },
  { label: 'FEAR (3)', value: 'Fear' },
  { label: 'DISGUST (2)', value: 'Disgust' },
  { label: 'ANGRY (3)', value: 'Angry' },
];

export const SampleGallery: React.FC<SampleGalleryProps> = ({
  selectedSampleId,
  onSelectSample,
  prediction,
  isEvaluating,
  evaluatedSamples,
  onRunAllBenchmarks,
  isBatchEvaluating = false,
  batchProgress = null,
}) => {
  const [activeCategory, setActiveCategory] = useState<'ALL' | ExpressionType>('ALL');

  const filteredSamples = activeCategory === 'ALL'
    ? SAMPLE_IMAGES
    : SAMPLE_IMAGES.filter((s) => s.expectedEmotion === activeCategory);

  const currentSample = SAMPLE_IMAGES.find((s) => s.id === selectedSampleId) || SAMPLE_IMAGES[0];

  // Dynamically calculate metrics strictly from real backend prediction history
  const totalSamples = SAMPLE_IMAGES.length;
  const evaluatedCount = Object.keys(evaluatedSamples).length;
  const matchCount = Object.values(evaluatedSamples).filter((e) => e.match).length;
  const hasEvaluations = evaluatedCount > 0;
  const accuracyPct = hasEvaluations ? ((matchCount / evaluatedCount) * 100).toFixed(1) : null;

  return (
    <div className="w-full rounded-xl bg-slate-900 border border-slate-800 p-4 sm:p-5 shadow-lg space-y-4">
      {/* 1. Header with Dynamic Evaluation Summary */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3.5 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-4 h-4" />
              REFERENCE SAMPLE EVALUATION
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950 text-slate-300 border border-slate-800">
              BENCHMARK SET
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Transparent comparison of labeled reference targets vs authoritative A4 model predictions.
          </p>
        </div>

        {/* Benchmark All Button */}
        {onRunAllBenchmarks && (
          <button
            onClick={onRunAllBenchmarks}
            disabled={isBatchEvaluating || isEvaluating}
            className={`min-h-[38px] px-3.5 py-1.5 rounded-lg font-mono text-xs font-semibold transition-all flex items-center gap-2 shrink-0 active:scale-95 ${
              isBatchEvaluating
                ? 'bg-cyan-950 text-cyan-400 border border-cyan-800 cursor-wait'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 border border-cyan-400 shadow-sm'
            }`}
          >
            {isBatchEvaluating ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>BENCHMARKING ({batchProgress ? `${batchProgress.current}/${batchProgress.total}` : 'RUNNING'})...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>EVALUATE ALL 20 SAMPLES</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* 2. Dynamic Metric Summary Cards */}
      <div className="grid grid-cols-3 gap-2.5 sm:gap-3 font-mono text-xs">
        {/* Metric 1: Reference Set */}
        <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-[10px] text-slate-400 uppercase block tracking-wider font-medium">
            REFERENCE SET
          </span>
          <span className="text-base sm:text-lg font-bold text-slate-200 mt-0.5 block">
            {totalSamples} SAMPLES
          </span>
          <span className="text-[9px] text-slate-500 block">7 Canonical Classes</span>
        </div>

        {/* Metric 2: Matches */}
        <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-[10px] text-slate-400 uppercase block tracking-wider font-medium">
            MATCHES
          </span>
          <span className="text-base sm:text-lg font-bold mt-0.5 block">
            {hasEvaluations ? (
              <span className="text-emerald-400">
                {matchCount} / {evaluatedCount}
              </span>
            ) : (
              <span className="text-slate-500 text-xs sm:text-sm">NOT EVALUATED</span>
            )}
          </span>
          <span className="text-[9px] text-slate-500 block">
            {hasEvaluations ? `${evaluatedCount} Evaluated` : 'Click Sample to Test'}
          </span>
        </div>

        {/* Metric 3: Reference Accuracy */}
        <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-center">
          <span className="text-[10px] text-slate-400 uppercase block tracking-wider font-medium">
            REFERENCE ACCURACY
          </span>
          <span className="text-base sm:text-lg font-bold mt-0.5 block">
            {hasEvaluations ? (
              <span className="text-cyan-400">{accuracyPct}%</span>
            ) : (
              <span className="text-slate-500 text-xs sm:text-sm">NOT EVALUATED</span>
            )}
          </span>
          <span className="text-[9px] text-slate-500 block">
            {hasEvaluations ? 'Real Model Rate' : 'Awaiting Run'}
          </span>
        </div>
      </div>

      {/* 3. Category Filter Chips */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-slate-800">
        {CATEGORIES.map((cat) => {
          const isActive = activeCategory === cat.value;
          return (
            <button
              key={cat.value}
              onClick={() => setActiveCategory(cat.value)}
              className={`min-h-[34px] px-3 py-1 rounded-md text-xs font-mono font-medium transition-all shrink-0 whitespace-nowrap active:scale-95 ${
                isActive
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                  : 'bg-slate-950/80 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* 4. Improved Sample Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {filteredSamples.map((sample) => {
          const isSelected = sample.id === selectedSampleId;
          const refEmoji = EXPRESSION_EMOJIS[sample.expectedEmotion] || '';
          const evalRecord = evaluatedSamples[sample.id];

          return (
            <button
              key={sample.id}
              onClick={() => onSelectSample(sample)}
              className={`group relative rounded-lg overflow-hidden border text-left transition-all flex flex-col active:scale-98 min-h-[44px] ${
                isSelected
                  ? 'border-cyan-400 bg-cyan-950/30 ring-2 ring-cyan-400/90 shadow-md'
                  : 'border-slate-800 bg-slate-950 hover:border-slate-700 hover:bg-slate-900/60'
              }`}
            >
              {/* Thumbnail Container */}
              <div className="relative w-full aspect-square bg-slate-950 overflow-hidden">
                <img
                  src={sample.image}
                  alt={`${sample.title} - Reference: ${sample.expectedEmotion}`}
                  loading="lazy"
                  className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-103"
                />

                {/* ID Pill and Active State */}
                <div className="absolute top-1.5 left-1.5 right-1.5 flex items-center justify-between pointer-events-none">
                  <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-slate-950/90 text-slate-300 border border-slate-800/80">
                    {sample.id}
                  </span>
                  {isSelected && (
                    <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-cyan-400 text-slate-950 font-bold shadow-sm">
                      SELECTED
                    </span>
                  )}
                </div>

                {/* Live/Recorded Evaluation Dot */}
                {evalRecord && (
                  <div className="absolute bottom-1.5 right-1.5 pointer-events-none">
                    {evalRecord.match ? (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-emerald-950/90 border border-emerald-500/50 text-emerald-300 font-mono text-[9px] font-semibold">
                        <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                        MATCH
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-amber-950/90 border border-amber-500/50 text-amber-300 font-mono text-[9px] font-semibold">
                        <AlertTriangle className="w-2.5 h-2.5 text-amber-400" />
                        {evalRecord.predicted.toUpperCase()}
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Card Footer: Clear Reference Label */}
              <div className="p-2 sm:p-2.5 w-full flex flex-col gap-0.5 border-t border-slate-800/80 bg-slate-950">
                <span className="text-[9px] font-mono text-slate-400 uppercase tracking-wider font-semibold">
                  REFERENCE
                </span>
                <div className="flex items-center gap-1.5 truncate">
                  <span className="text-sm shrink-0">{refEmoji}</span>
                  <span className={`text-xs font-bold tracking-wide truncate uppercase ${
                    isSelected ? 'text-cyan-300' : 'text-slate-100'
                  }`}>
                    {sample.expectedEmotion}
                  </span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* 5. Scientific Integrity Notice */}
      <div className="text-[11px] text-slate-500 font-mono text-center sm:text-left pt-1">
        Ground-truth labels reflect original reference dataset annotations. All model predictions are executed live by the authoritative A4 pipeline without overrides or synthesis.
      </div>
    </div>
  );
};
