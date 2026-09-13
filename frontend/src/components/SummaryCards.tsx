import React from 'react';
import { AnalysisResult } from '../types/eeg';

interface SummaryCardsProps {
  result: AnalysisResult;
  threshold: number;
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({ result, threshold }) => {
  const flaggedRate = result.total_segments > 0
    ? ((result.seizure_segments / result.total_segments) * 100).toFixed(2)
    : '0.00';

  const stats = [
    {
      label: 'Segments analyzed',
      value: result.total_segments.toLocaleString(),
      subtext: `${(result.duration_seconds / 60).toFixed(1)} min total duration`,
    },
    {
      label: 'Flagged segments',
      value: result.seizure_segments.toLocaleString(),
      subtext: result.seizure_segments > 0 ? 'Exceeded threshold' : 'Zero segments flagged',
      highlight: result.seizure_segments > 0 ? 'text-rose-600' : 'text-slate-900',
    },
    {
      label: 'Flagged segment rate',
      value: `${flaggedRate}%`,
      subtext: 'Proportion of recording',
    },
    {
      label: 'Decision threshold',
      value: threshold.toFixed(2),
      subtext: 'VQC probability cutoff',
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, idx) => (
        <div
          key={idx}
          className="surface-card rounded-xl p-5 hover:border-slate-300 transition-colors"
        >
          <span className="text-xs font-medium text-slate-500 font-mono uppercase tracking-wider">
            {stat.label}
          </span>
          <p
            className={`text-2xl sm:text-3xl font-extrabold font-mono tracking-tight mt-1 ${
              stat.highlight || 'text-slate-900'
            }`}
          >
            {stat.value}
          </p>
          <p className="text-[11px] text-slate-400 mt-1">
            {stat.subtext}
          </p>
        </div>
      ))}
    </div>
  );
};
