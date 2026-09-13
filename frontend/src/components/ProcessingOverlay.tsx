import React, { useEffect, useState } from 'react';
import { Activity, Check, Loader2 } from 'lucide-react';

export const ProcessingOverlay: React.FC = () => {
  const [activeStepIndex, setActiveStepIndex] = useState(0);

  const steps = [
    { name: 'EEG recording loaded', desc: 'Validating EDF headers & 256 Hz sampling frequency' },
    { name: 'Preprocessing & Filtering', desc: 'Applying 0.5–40 Hz Butterworth bandpass filter' },
    { name: 'Feature Extraction', desc: 'Computing wavelet energy, entropy & variance (20 features)' },
    { name: 'PCA Transformation', desc: 'Projecting 20 features to 4 principal components' },
    { name: 'VQC Quantum Inference', desc: 'Evaluating ZZFeatureMap & RealAmplitudes statevector' },
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setActiveStepIndex(1), 1200);
    const timer2 = setTimeout(() => setActiveStepIndex(2), 2600);
    const timer3 = setTimeout(() => setActiveStepIndex(3), 4200);
    const timer4 = setTimeout(() => setActiveStepIndex(4), 5800);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, []);

  return (
    <div className="py-20 flex items-center justify-center">
      <div className="surface-card rounded-2xl p-8 sm:p-12 max-w-lg w-full shadow-premium text-center">
        
        {/* Animated Icon Indicator */}
        <div className="w-12 h-12 rounded-xl bg-sky-50 text-sky-600 border border-sky-200 flex items-center justify-center mx-auto mb-6">
          <Loader2 className="w-6 h-6 animate-spin text-sky-600" />
        </div>

        <h3 className="text-2xl font-bold text-slate-900 tracking-tight">
          Analyzing EEG
        </h3>
        <p className="text-sm text-slate-500 mt-1 mb-8">
          Processing your recording through the quantum EEG pipeline.
        </p>

        {/* Step Progress Checklist */}
        <div className="space-y-4 text-left border-t border-slate-100 pt-6">
          {steps.map((step, idx) => {
            const isCompleted = idx < activeStepIndex;
            const isCurrent = idx === activeStepIndex;

            return (
              <div key={idx} className="flex items-start gap-3">
                <div className="mt-0.5 shrink-0">
                  {isCompleted ? (
                    <div className="w-5 h-5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200 flex items-center justify-center">
                      <Check className="w-3 h-3 stroke-[3]" />
                    </div>
                  ) : isCurrent ? (
                    <div className="w-5 h-5 rounded-full bg-sky-50 text-sky-600 border border-sky-300 flex items-center justify-center">
                      <span className="w-2 h-2 rounded-full bg-sky-600 animate-ping" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 rounded-full bg-slate-100 border border-slate-200" />
                  )}
                </div>

                <div>
                  <p
                    className={`text-sm font-medium ${
                      isCompleted
                        ? 'text-slate-700'
                        : isCurrent
                        ? 'text-sky-700 font-semibold'
                        : 'text-slate-400'
                    }`}
                  >
                    {step.name}
                  </p>
                  <p className="text-xs text-slate-400">
                    {step.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        <p className="text-[11px] text-slate-400 font-mono mt-8">
          Quantum simulation running on Qiskit backend • Please wait...
        </p>

      </div>
    </div>
  );
};
