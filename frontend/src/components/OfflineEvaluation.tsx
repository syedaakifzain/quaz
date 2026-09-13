import React from 'react';
import { Info, BarChart3 } from 'lucide-react';

export const OfflineEvaluation: React.FC = () => {
  const metrics = [
    { label: 'Accuracy', value: '91.19%', sub: 'Overall proportion of correctly classified segments' },
    { label: 'Specificity', value: '92.53%', sub: 'Proportion of non-seizure segments correctly identified' },
    { label: 'Recall', value: '18.29%', sub: 'Proportion of seizure segments detected' },
    { label: 'Precision', value: '3.77%', sub: 'Proportion of predicted seizure segments that were correct' },
    { label: 'F1 Score', value: '6.16%', sub: 'Harmonic mean of precision and recall' },
  ];

  return (
    <section id="offline-evaluation" className="py-20 bg-slate-100/50 border-t border-slate-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="max-w-2xl mb-10">
          <p className="text-xs font-mono font-semibold tracking-wider text-sky-600 uppercase mb-1">
            Model Validation
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Offline model validation
          </h2>
          <p className="text-sm text-slate-600 mt-2">
            Offline evaluation results from subject-aware nested Leave-One-Subject-Out (LOSO) validation on the CHB-MIT dataset. These metrics describe the trained model's performance on held-out subjects and are independent of the recording being analyzed.          </p>
        </div>

        {/* Metric Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {metrics.map((m, idx) => (
            <div
              key={idx}
              className="surface-card rounded-xl p-5 hover:border-slate-300 transition-colors"
            >
              <span className="text-xs font-medium text-slate-500 uppercase font-mono">
                {m.label}
              </span>
              <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono tracking-tight mt-1">
                {m.value}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">
                {m.sub}
              </p>
            </div>
          ))}
        </div>

        {/* Scientific Honesty Disclaimer Note */}
        <div className="mt-6 flex items-start gap-3 p-4 rounded-xl bg-white border border-slate-200/80 text-xs text-slate-600">
          <Info className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Scientific Integrity Note:</strong> Metrics shown here are from offline nested LOSO evaluation on subjects chb01–chb05 and do not represent performance on the currently uploaded recording. The current model demonstrates higher specificity (92.53%) than seizure recall (18.29%), indicating that non-seizure segments are identified more reliably than seizure segments. These results are presented transparently as the current evaluation baseline for further model improvement.
          </p>
        </div>

      </div>
    </section>
  );
};
