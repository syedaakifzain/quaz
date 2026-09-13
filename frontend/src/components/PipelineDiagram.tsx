import React from 'react';
import { ArrowRight, GitCommit, FileCode, Sliders, Cpu, Activity } from 'lucide-react';

export const PipelineDiagram: React.FC = () => {
  const steps = [
    { title: 'EDF Input', badge: 'Signal', icon: FileCode },
    { title: 'Preprocessing', badge: '0.5-40 Hz', icon: Sliders },
    { title: 'Segmentation', badge: '2s Windows', icon: Activity },
    { title: 'Feature Extractor', badge: '20 Features', icon: GitCommit },
    { title: 'StandardScaler', badge: 'Z-score', icon: Sliders },
    { title: 'PCA', badge: '4D Reduction', icon: GitCommit },
    { title: 'MinMaxScaler', badge: '[0, 1] Range', icon: Sliders },
    { title: 'VQC Engine', badge: '4 Qubits', icon: Cpu },
    { title: 'Probability', badge: 'Threshold 0.60', icon: Activity },
  ];

  return (
    <div className="glass-panel p-6 rounded-2xl border border-slate-800 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
          End-to-End Inference Pipeline Diagram
        </h3>
        <span className="text-xs font-mono text-cyan-400 bg-cyan-950 px-2.5 py-1 rounded-md border border-cyan-500/30">
          InferenceService.predict()
        </span>
      </div>

      <div className="overflow-x-auto py-2">
        <div className="flex items-center gap-2 min-w-[900px] justify-between">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isLast = idx === steps.length - 1;
            const isQuantum = step.title === 'VQC Engine';

            return (
              <React.Fragment key={idx}>
                <div
                  className={`flex flex-col items-center p-3 rounded-xl border text-center transition-all ${
                    isQuantum
                      ? 'bg-cyan-950/60 border-cyan-500/60 shadow-quantum-glow ring-1 ring-cyan-400'
                      : isLast
                      ? 'bg-rose-950/40 border-rose-500/40'
                      : 'bg-navy-950 border-slate-800'
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center mb-1.5 ${
                      isQuantum
                        ? 'bg-cyan-900 text-cyan-300'
                        : isLast
                        ? 'bg-rose-900 text-rose-300'
                        : 'bg-slate-900 text-slate-400'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-bold font-mono text-slate-200 block truncate max-w-[90px]">
                    {step.title}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400 block mt-0.5">
                    {step.badge}
                  </span>
                </div>

                {!isLast && (
                  <ArrowRight className="w-4 h-4 text-slate-600 shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
};
