import React, { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Filter } from 'lucide-react';
import { SegmentPrediction } from '../types/eeg';

interface DetectionTableProps {
  segmentPredictions: SegmentPrediction[];
  threshold: number;
  selectedSegmentIndex: number | null;
  onSelectSegment: (segment: SegmentPrediction) => void;
}

export const DetectionTable: React.FC<DetectionTableProps> = ({
  segmentPredictions,
  threshold,
  selectedSegmentIndex,
  onSelectSegment,
}) => {
  const [filterMode, setFilterMode] = useState<'ALL' | 'FLAGGED'>('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  const filteredList = useMemo(() => {
    if (filterMode === 'FLAGGED') {
      return segmentPredictions.filter((s) => s.seizure_probability >= threshold);
    }
    return segmentPredictions;
  }, [segmentPredictions, filterMode, threshold]);

  const totalPages = Math.max(1, Math.ceil(filteredList.length / pageSize));
  const pageSegments = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredList.slice(start, start + pageSize);
  }, [filteredList, currentPage]);

  const handleFilterChange = (mode: 'ALL' | 'FLAGGED') => {
    setFilterMode(mode);
    setCurrentPage(1);
  };

  return (
    <div className="surface-card rounded-2xl p-6 sm:p-8 shadow-premium">
      
      {/* Table Header & Filter Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
        <div>
          <h4 className="text-lg font-bold text-slate-900 tracking-tight">
            Segment classifications
          </h4>
          <p className="text-xs text-slate-500 mt-0.5">
            Showing {filteredList.length} segments ({filterMode === 'FLAGGED' ? 'Flagged only' : 'All'})
          </p>
        </div>

        {/* Filter Toggle */}
        <div className="flex items-center gap-1 p-1 rounded-lg bg-slate-100 text-xs font-medium">
          <button
            onClick={() => handleFilterChange('ALL')}
            className={`px-3 py-1.5 rounded-md transition-all ${
              filterMode === 'ALL'
                ? 'bg-white text-slate-900 shadow-subtle'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            All ({segmentPredictions.length})
          </button>
          <button
            onClick={() => handleFilterChange('FLAGGED')}
            className={`px-3 py-1.5 rounded-md transition-all ${
              filterMode === 'FLAGGED'
                ? 'bg-white text-rose-700 shadow-subtle font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Flagged ({segmentPredictions.filter((s) => s.seizure_probability >= threshold).length})
          </button>
        </div>
      </div>

      {/* Minimal Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-200/80 text-slate-500 uppercase tracking-wider">
              <th className="py-3 px-4 font-semibold">Segment</th>
              <th className="py-3 px-4 font-semibold">Time Window</th>
              <th className="py-3 px-4 font-semibold">Duration</th>
              <th className="py-3 px-4 font-semibold">Probability</th>
              <th className="py-3 px-4 font-semibold">Model Prediction</th>
              <th className="py-3 px-4 font-semibold text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {pageSegments.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-400">
                  No segments match the selected filter.
                </td>
              </tr>
            ) : (
              pageSegments.map((seg) => {
                const isFlagged = seg.seizure_probability >= threshold;
                const isSelected = seg.segment_index === selectedSegmentIndex;

                return (
                  <tr
                    key={seg.segment_index}
                    onClick={() => onSelectSegment(seg)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-sky-50/80'
                        : isFlagged
                        ? 'bg-rose-50/40 hover:bg-rose-50/70'
                        : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-3.5 px-4 font-medium text-slate-900">
                      #{seg.segment_index}
                    </td>
                    <td className="py-3.5 px-4 text-slate-600">
                      {seg.start_time_seconds}s – {seg.end_time_seconds}s
                    </td>
                    <td className="py-3.5 px-4 text-slate-500">
                      {seg.end_time_seconds - seg.start_time_seconds}s
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`font-semibold ${isFlagged ? 'text-rose-600' : 'text-slate-700'}`}>
                        {(seg.seizure_probability * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-700 font-sans">
                      {isFlagged ? 'Seizure' : 'Non-Seizure'}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                          isFlagged
                            ? 'bg-rose-100 text-rose-700'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {isFlagged ? 'FLAGGED' : 'NORMAL'}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex items-center justify-between pt-4 mt-2 border-t border-slate-100 text-xs text-slate-500 font-mono">
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <div className="flex items-center gap-2">
          <button
            disabled={currentPage <= 1}
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            disabled={currentPage >= totalPages}
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

    </div>
  );
};
