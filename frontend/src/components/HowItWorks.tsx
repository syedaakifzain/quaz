import React from 'react';
import { FileText, Filter, BarChart2, Minimize2, Cpu, Activity, ArrowRight } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const steps = [
    {
      num: '01',
      title: 'EEG Recording',
      desc: 'Raw European Data Format (.edf) signal ingestion',
      icon: FileText,
    },
    {
      num: '02',
      title: 'Preprocessing',
      desc: 'Bandpass filtering (0.5–40 Hz) & channel validation',
      icon: Filter,
    },
    {
      num: '03',
      title: '20 EEG Features',
      desc: 'Variance, spectral entropy & wavelet-energy extraction',
      icon: BarChart2,
    },
    {
      num: '04',
      title: 'PCA → 4D',
      desc: 'Orthogonal dimensionality reduction for quantum mapping',
      icon: Minimize2,
    },
    {
      num: '05',
      title: 'Quantum Circuit',
      desc: 'ZZFeatureMap encoding + RealAmplitudes ansatz',
      icon: Cpu,
    },
    {
      num: '06',
      title: 'Probability Signal',
      desc: 'Decision threshold evaluation at 0.60',
      icon: Activity,
    },
  ];

  return (
    <section id="how-it-works" className="py-24 sm:py-32">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="max-w-3xl mb-16 sm:mb-20">
          <p className="text-xs font-mono font-semibold tracking-wider text-sky-600 uppercase mb-2">
            Inference Pipeline
          </p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            From raw EEG to Quantum-Enhanced Probability Estimates.
          </h2>
          <p className="text-lg text-slate-600 mt-3 font-normal">
            Every recording follows the same preprocessing and feature pipeline used to train the V1.3.0 model.
          </p>
        </div>

        {/* Pipeline Steps (Horizontal on desktop, Vertical on mobile) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8 relative">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div key={idx} className="relative flex flex-col items-start group">

                {/* Step number badge & icon */}
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-slate-100 group-hover:bg-sky-50 border border-slate-200/80 group-hover:border-sky-200 flex items-center justify-center text-slate-700 group-hover:text-sky-600 transition-colors">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-mono font-bold text-slate-400 group-hover:text-sky-600 transition-colors">
                    {step.num}
                  </span>
                </div>

                {/* Content */}
                <h3 className="text-base font-semibold text-slate-900 tracking-tight mb-1">
                  {step.title}
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  {step.desc}
                </p>

                {/* Arrow indicator between steps (desktop) */}
                {idx < steps.length - 1 && (
                  <div className="hidden lg:block absolute -right-4 top-5 text-slate-300">
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                )}

              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
};
