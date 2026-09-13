import React from 'react';
import { Sliders, Clock, Activity, ShieldCheck, AlertTriangle } from 'lucide-react';
import { SegmentPrediction } from '../types/eeg';

interface SelectedSegmentProps {
  segment: SegmentPrediction | null;
  threshold: number;
}

export const SelectedSegment: React.FC<SelectedSegmentProps> = ({ segment, threshold }) => {
  if (!segment) {
    return (
      <div className="p-6 rounded-2xl border border-dashed border-slate-300 bg-slate-50/60 text-center text-slate-500 text-xs font-mono">
        Click any point on the chart or timeline above to inspect individual segment properties.
      </div>
    );
  }

  const isFlagged = segment.seizure_probability >= threshold;

  return (
    <div className="surface-card rounded-2xl p-6 sm:p-8 shadow-premium">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 mb-6 border-b border-slate-100">
        <div>
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-sky-600">
            Segment Details
          </span>
          <h4 className="text-lg font-bold text-slate-900 tracking-tight mt-0.5">
            Segment #{segment.segment_index} ({segment.start_time_seconds}s – {segment.end_time_seconds}s)
          </h4>
        </div>

        {/* Status Tag */}
        <span
          className={`px-3 py-1 rounded-full text-xs font-mono font-semibold inline-flex items-center gap-1.5 ${
            isFlagged
              ? 'bg-rose-50 text-rose-700 border border-rose-200'
              : 'bg-slate-100 text-slate-700 border border-slate-200'
          }`}
        >
          {isFlagged ? (
            <>
              <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
              STATUS: FLAGGED
            </>
          ) : (
            <>
              <ShieldCheck className="w-3.5 h-3.5 text-slate-500" />
              STATUS: NORMAL
            </>
          )}
        </span>
      </div>

      {/* 4 Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-slate-500">Probability</span>
          <p className={`text-xl font-bold mt-1 ${isFlagged ? 'text-rose-600' : 'text-slate-900'}`}>
            {(segment.seizure_probability * 100).toFixed(2)}%
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-slate-500">Threshold</span>
          <p className="text-xl font-bold text-slate-900 mt-1">
            {(threshold * 100).toFixed(0)}%
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-slate-500">Model Prediction</span>
          <p className="text-xl font-bold text-slate-900 mt-1">
            {isFlagged ? 'Seizure' : 'Non-Seizure'}
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-slate-500">Window Length</span>
          <p className="text-xl font-bold text-slate-900 mt-1">
            {segment.end_time_seconds - segment.start_time_seconds}s
          </p>
        </div>
      </div>
    </div>
  );
};
