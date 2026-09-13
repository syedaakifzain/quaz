import React from 'react';
import { SegmentPrediction } from '../types/eeg';

interface DetectionTimelineProps {
  segmentPredictions: SegmentPrediction[];
  threshold: number;
  selectedSegmentIndex: number | null;
  onSelectSegment: (segment: SegmentPrediction) => void;
}

export const DetectionTimeline: React.FC<DetectionTimelineProps> = ({
  segmentPredictions,
  threshold,
  selectedSegmentIndex,
  onSelectSegment,
}) => {
  const totalDuration = segmentPredictions.length > 0
    ? segmentPredictions[segmentPredictions.length - 1].end_time_seconds
    : 0;

  return (
    <div className="bg-[#faf7ed] rounded-2xl p-6 sm:p-8 shadow-premium">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-bold text-slate-900">Recording timeline</h4>
          <p className="text-xs text-slate-500">
            Continuous distribution of 2-second windows across {Math.floor(totalDuration / 60)} minutes
          </p>
        </div>
        <span className="text-xs font-mono text-slate-500">
          {segmentPredictions.length} total segments
        </span>
      </div>

      {/* Interactive Micro-timeline Bar */}
      <div className="relative h-7 w-full bg-slate-100 rounded-lg overflow-hidden flex items-stretch border border-slate-200 cursor-pointer">
        {segmentPredictions.map((seg) => {
          const isFlagged = seg.seizure_probability >= threshold;
          const isSelected = seg.segment_index === selectedSegmentIndex;

          return (
            <div
              key={seg.segment_index}
              onClick={() => onSelectSegment(seg)}
              title={`Segment #${seg.segment_index} (${seg.start_time_seconds}s–${seg.end_time_seconds}s): ${(seg.seizure_probability * 100).toFixed(1)}%`}
              className={`flex-1 transition-all ${isSelected
                ? 'bg-sky-600 z-10 scale-y-125'
                : isFlagged
                  ? 'bg-rose-500 hover:bg-rose-600'
                  : 'hover:bg-slate-300'
                }`}
            />
          );
        })}
      </div>

      {/* Axis markers */}
      <div className="flex justify-between items-center text-[11px] font-mono text-slate-400 mt-2">
        <span>00:00</span>
        <span>{Math.floor((totalDuration / 2) / 60)}:00</span>
        <span>{Math.floor(totalDuration / 60)}:00</span>
      </div>
    </div>
  );
};
