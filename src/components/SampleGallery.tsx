import React, { useState } from 'react';
import { SAMPLE_IMAGES, SampleImageItem } from '../data/sampleImages';
import { ExpressionType, PredictResponse, EXPRESSION_EMOJIS } from '../services/api';
import { CheckCircle2, AlertCircle, Sparkles, Filter, RefreshCw, Eye } from 'lucide-react';

interface SampleGalleryProps {
  selectedSampleId: string;
  onSelectSample: (sample: SampleImageItem) => void;
  prediction: PredictResponse | null;
  isEvaluating: boolean;
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
}) => {
  const [activeCategory, setActiveCategory] = useState<'ALL' | ExpressionType>('ALL');

  const filteredSamples = activeCategory === 'ALL'
    ? SAMPLE_IMAGES
    : SAMPLE_IMAGES.filter((s) => s.expectedEmotion === activeCategory);

  const currentSample = SAMPLE_IMAGES.find((s) => s.id === selectedSampleId) || SAMPLE_IMAGES[0];

  // Compare expected emotion against real model prediction
  const isMatch = prediction && currentSample && prediction.face_detected && (
    prediction.prediction.toLowerCase() === currentSample.expectedEmotion.toLowerCase()
  );

  return (
    <div className="w-full rounded-2xl bg-slate-900/90 border border-slate-800 p-4 sm:p-5 backdrop-blur-xl shadow-xl space-y-5">
      {/* 1. Header with Evaluation Philosophy */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-4 h-4" />
              REFERENCE SAMPLE GALLERY
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800">
              20 BENCHMARK SAMPLES
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-sans">
            Evaluate the real A4 model with genuine labeled portrait samples across seven canonical expressions.
          </p>
        </div>

        {/* Live Evaluation Status Pill */}
        <div className="flex items-center gap-2 self-start sm:self-auto font-mono text-[11px]">
          <span className="text-slate-400">MODEL EVALUATION:</span>
          {isEvaluating ? (
            <span className="inline-flex items-center gap-1 text-cyan-400 font-semibold">
              <RefreshCw className="w-3 h-3 animate-spin" />
              INFERRING...
            </span>
          ) : isMatch ? (
            <span className="inline-flex items-center gap-1 text-emerald-400 font-bold bg-emerald-950/70 border border-emerald-500/40 px-2 py-0.5 rounded">
              <CheckCircle2 className="w-3 h-3" />
              ✓ MODEL MATCH
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-amber-400 font-semibold bg-amber-950/70 border border-amber-500/40 px-2 py-0.5 rounded">
              <AlertCircle className="w-3 h-3" />
              MODEL DIFFERENCE
            </span>
          )}
        </div>
      </div>

      {/* 2. Active Sample Evaluation Inspector Card */}
      {currentSample && (
        <div className="p-3.5 sm:p-4 rounded-xl bg-slate-950/90 border border-slate-800/90 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-xl overflow-hidden border border-slate-700 shrink-0 relative bg-slate-900">
              <img
                src={currentSample.image}
                alt={currentSample.title}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <span className="absolute bottom-0 inset-x-0 bg-slate-950/80 text-[9px] font-mono text-center text-slate-300 py-0.5 truncate">
                {currentSample.id}
              </span>
            </div>
            <div className="space-y-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h4 className="text-sm font-bold text-white tracking-tight truncate">
                  {currentSample.title}
                </h4>
                <span className="text-[10px] font-mono text-slate-400">
                  {currentSample.attribution}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed font-sans line-clamp-2">
                {currentSample.description}
              </p>
            </div>
          </div>

          {/* Model vs Expected Comparison Metrics */}
          <div className="w-full md:w-auto grid grid-cols-3 gap-2 sm:gap-3 font-mono text-xs shrink-0">
            <div className="p-2 sm:p-2.5 rounded-lg bg-slate-900/80 border border-slate-800/80 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">EXPECTED</span>
              <span className="font-bold text-slate-200 flex items-center justify-center gap-1 mt-0.5">
                <span>{EXPRESSION_EMOJIS[currentSample.expectedEmotion] || ''}</span>
                <span>{currentSample.expectedEmotion}</span>
              </span>
            </div>

            <div className="p-2 sm:p-2.5 rounded-lg bg-slate-900/80 border border-slate-800/80 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">PREDICTION</span>
              <span className={`font-bold flex items-center justify-center gap-1 mt-0.5 ${
                isMatch ? 'text-emerald-400' : 'text-amber-300'
              }`}>
                <span>{prediction ? (EXPRESSION_EMOJIS[prediction.prediction] || '') : '—'}</span>
                <span>{prediction?.prediction || '—'}</span>
              </span>
            </div>

            <div className="p-2 sm:p-2.5 rounded-lg bg-slate-900/80 border border-slate-800/80 text-center">
              <span className="text-[10px] text-slate-400 block uppercase">CONFIDENCE</span>
              <span className="font-bold text-cyan-400 mt-0.5 block">
                {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '0.0%'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 3. Category Filter Chips (Horizontally Scrollable on Mobile) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-mono text-[11px] flex items-center gap-1.5 uppercase font-medium">
            <Filter className="w-3 h-3 text-cyan-400" />
            FILTER BY EXPRESSION
          </span>
          <span className="text-[11px] font-mono">
            Showing {filteredSamples.length} of {SAMPLE_IMAGES.length}
          </span>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto pb-1.5 scrollbar-thin scrollbar-thumb-slate-800 -mx-1 px-1">
          {CATEGORIES.map((cat) => {
            const isActive = activeCategory === cat.value;
            return (
              <button
                key={cat.value}
                onClick={() => setActiveCategory(cat.value)}
                className={`min-h-[40px] px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all shrink-0 whitespace-nowrap active:scale-95 focus-visible:outline-2 focus-visible:outline-cyan-400 ${
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
      </div>

      {/* 4. Responsive Sample Thumbnails Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 sm:gap-3.5">
        {filteredSamples.map((sample) => {
          const isSelected = sample.id === selectedSampleId;
          const emoji = EXPRESSION_EMOJIS[sample.expectedEmotion] || '';

          return (
            <button
              key={sample.id}
              onClick={() => onSelectSample(sample)}
              className={`group relative rounded-xl overflow-hidden border text-left transition-all flex flex-col focus-visible:outline-2 focus-visible:outline-cyan-400 active:scale-98 min-h-[44px] ${
                isSelected
                  ? 'border-cyan-400 bg-cyan-950/40 ring-2 ring-cyan-400/80 shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)]'
                  : 'border-slate-800 bg-slate-950 hover:border-slate-700 hover:bg-slate-900/60'
              }`}
            >
              {/* Thumbnail Container with 1:1 Aspect Ratio */}
              <div className="relative w-full aspect-square bg-slate-950 overflow-hidden">
                <img
                  src={sample.image}
                  alt={`${sample.title} - Expected: ${sample.expectedEmotion}`}
                  loading="lazy"
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                />

                {/* Top Badge Overlay */}
                <div className="absolute top-1.5 left-1.5 right-1.5 flex items-center justify-between pointer-events-none">
                  <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-slate-950/85 text-slate-300 backdrop-blur-sm border border-slate-800/80">
                    {sample.id}
                  </span>
                  {isSelected && (
                    <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-cyan-400 text-slate-950 font-bold shadow-sm">
                      ACTIVE
                    </span>
                  )}
                </div>

                {/* Selection Overlay */}
                <div className={`absolute inset-0 bg-cyan-500/10 transition-opacity ${
                  isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                }`} />
              </div>

              {/* Card Footer: Expected Expression Label */}
              <div className="p-2 sm:p-2.5 w-full flex items-center justify-between border-t border-slate-800/80">
                <div className="min-w-0 pr-1">
                  <span className="text-[10px] font-mono text-slate-400 block uppercase truncate">
                    EXPECTED
                  </span>
                  <span className={`text-xs font-bold tracking-wide flex items-center gap-1 truncate ${
                    isSelected ? 'text-cyan-300' : 'text-white'
                  }`}>
                    <span>{emoji}</span>
                    <span className="truncate">{sample.expectedEmotion.toUpperCase()}</span>
                  </span>
                </div>
                <div className="w-5 h-5 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 shrink-0">
                  <Eye className="w-3 h-3" />
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* 5. Testing Notice */}
      <div className="pt-1 text-[11px] text-slate-500 font-mono text-center sm:text-left">
        Note: Selecting any sample image streams its raw pixel buffer directly through the authoritative A4 model pipeline (/predict). No outputs or probabilities are synthesized or hardcoded.
      </div>
    </div>
  );
};
