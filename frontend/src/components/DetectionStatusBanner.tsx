import React from 'react';
import { AlertCircle, CheckCircle2, Sliders } from 'lucide-react';
import { AnalysisResult } from '../types/eeg';

interface DetectionStatusBannerProps {
  result: AnalysisResult;
  threshold: number;
}

export const DetectionStatusBanner: React.FC<DetectionStatusBannerProps> = ({ result, threshold }) => {
  const hasFlagged = result.seizure_segments > 0;

  return (
    <div
      className={`p-6 rounded-2xl border transition-all ${
        hasFlagged
          ? 'bg-rose-50/70 border-rose-200 text-rose-900'
          : 'bg-emerald-50/70 border-emerald-200 text-emerald-900'
      }`}
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        
        <div className="flex items-center gap-3.5">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
              hasFlagged
                ? 'bg-rose-100 text-rose-600 border border-rose-200'
                : 'bg-emerald-100 text-emerald-600 border border-emerald-200'
            }`}
          >
            {hasFlagged ? (
              <AlertCircle className="w-5 h-5" />
            ) : (
              <CheckCircle2 className="w-5 h-5" />
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight">
              {hasFlagged
                ? 'Possible seizure activity detected'
                : 'No significant activity flagged'}
            </h2>
            <p className="text-xs opacity-80 mt-0.5">
              {hasFlagged
                ? `${result.seizure_segments} segments exceeded the model decision threshold (${threshold.toFixed(2)}).`
                : `All ${result.total_segments} evaluated segments remained below the decision threshold (${threshold.toFixed(2)}).`}
            </p>
          </div>
        </div>

        {/* Threshold badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/80 border border-current text-xs font-mono font-medium shadow-subtle shrink-0">
          <Sliders className="w-3.5 h-3.5" />
          <span>Model threshold: {threshold.toFixed(2)}</span>
        </div>

      </div>
    </div>
  );
};
