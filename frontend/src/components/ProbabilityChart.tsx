import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { SegmentPrediction } from '../types/eeg';

interface ProbabilityChartProps {
  segmentPredictions: SegmentPrediction[];
  threshold: number;
  selectedSegmentIndex: number | null;
  onSelectSegment: (segment: SegmentPrediction) => void;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  segmentPredictions,
  threshold,
  selectedSegmentIndex,
  onSelectSegment,
}) => {
  // Sample data points for chart performance if large dataset (>600 points)
  const chartData = useMemo(() => {
    const total = segmentPredictions.length;
    const step = total > 600 ? Math.ceil(total / 600) : 1;

    return segmentPredictions
      .filter((_, idx) => idx % step === 0 || _.seizure_probability >= threshold)
      .map((seg) => ({
        index: seg.segment_index,
        time: seg.start_time_seconds,
        timeLabel: `${Math.floor(seg.start_time_seconds / 60)}m ${Math.floor(seg.start_time_seconds % 60)}s`,
        probability: Number((seg.seizure_probability * 100).toFixed(2)),
        rawProbability: seg.seizure_probability,
        isFlagged: seg.seizure_probability >= threshold,
        rawSegment: seg,
      }));
  }, [segmentPredictions, threshold]);

  return (
    <div className="bg-[#FAF7ED] rounded-2xl p-6 sm:p-8 shadow-premium">
      {/* Chart Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="text-lg font-bold text-slate-900 tracking-tight">
            Seizure probability over time
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Model probability for each 2-second EEG segment. Click any point to inspect details.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-sky-600 rounded" />
            <span className="text-slate-600">Probability Signal</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-rose-500 border-b border-dashed border-rose-500" />
            <span className="text-rose-700">Threshold ({(threshold * 100).toFixed(0)}%)</span>
          </div>
        </div>
      </div>

      {/* Main Chart Container */}
      <div className="w-full h-72 sm:h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
            onClick={(e) => {
              if (e && e.activePayload && e.activePayload[0]) {
                const seg = e.activePayload[0].payload.rawSegment;
                if (seg) onSelectSegment(seg);
              }
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
              tickFormatter={(time) => `${Math.floor(time / 60)}m`}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
              tickFormatter={(val) => `${val}%`}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-premium text-xs space-y-1 font-mono">
                      <p className="font-semibold text-slate-900">Time: {data.timeLabel} ({data.time}s)</p>
                      <p className="text-sky-600">Probability: {data.probability}%</p>
                      <p className={data.isFlagged ? 'text-rose-600 font-bold' : 'text-slate-500'}>
                        Prediction: {data.isFlagged ? 'Flagged (Seizure)' : 'Normal'}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine
              y={threshold * 100}
              stroke="#f43f5e"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
            <Line
              type="monotone"
              dataKey="probability"
              stroke="#0284c7"
              strokeWidth={1.8}
              dot={false}
              activeDot={{ r: 5, fill: '#0284c7', stroke: '#ffffff', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
};
