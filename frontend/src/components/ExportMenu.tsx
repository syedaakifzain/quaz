import React from 'react';
import { Download, FileSpreadsheet, FileJson, RefreshCw, FileText } from 'lucide-react';
import { AnalysisResult } from '../types/eeg';

interface ExportMenuProps {
  result: AnalysisResult;
  onReset: () => void;
}

export const ExportMenu: React.FC<ExportMenuProps> = ({ result, onReset }) => {
  const downloadCSV = () => {
    const headers = ['Segment_Index', 'Start_Seconds', 'End_Seconds', 'Seizure_Probability', 'Model_Prediction'];
    const rows = result.segment_predictions.map((seg) => [
      seg.segment_index,
      seg.start_time_seconds,
      seg.end_time_seconds,
      seg.seizure_probability.toFixed(4),
      seg.predicted_label === 1 ? 'Seizure' : 'Non-Seizure',
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${result.recording_name}_predictions.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadJSON = () => {
    const jsonContent = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', jsonContent);
    link.setAttribute('download', `${result.recording_name}_analysis.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="surface-card rounded-2xl p-6 sm:p-8 shadow-premium flex flex-col sm:flex-row items-center justify-between gap-4">
      <div>
        <h4 className="text-base font-bold text-slate-900">Export Analysis Data</h4>
        <p className="text-xs text-slate-500 mt-0.5">
          Download segment classifications or analyze another recording.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
        <button
          onClick={downloadCSV}
          className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium font-mono shadow-subtle hover:-translate-y-0.5 transition-all cursor-pointer"
        >
          <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600" />
          Export CSV
        </button>

        <button
          onClick={downloadJSON}
          className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium font-mono shadow-subtle hover:-translate-y-0.5 transition-all cursor-pointer"
        >
          <FileJson className="w-3.5 h-3.5 text-sky-600" />
          Export JSON
        </button>

        <button
          onClick={onReset}
          className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium shadow-sm hover:-translate-y-0.5 transition-all cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Analyze another recording
        </button>
      </div>
    </div>
  );
};
