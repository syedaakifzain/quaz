import React from 'react';

export const MetricsStrip: React.FC = () => {
  const metrics = [
    { value: '20', label: 'EEG Features' },
    { value: '4', label: 'Quantum Qubits' },
    { value: '2s', label: 'Analysis Windows' },
    { value: 'VQC', label: 'Classifier' },
  ];

  return (
    <section className="border-y border-slate-200/80 bg-white/60 py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4 divide-y md:divide-y-0 md:divide-x divide-slate-200/80">
          {metrics.map((m, idx) => (
            <div
              key={idx}
              className={`flex flex-col items-center justify-center text-center ${
                idx > 0 ? 'pt-6 md:pt-0' : ''
              }`}
            >
              <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight font-mono">
                {m.value}
              </span>
              <span className="text-sm font-medium text-slate-500 mt-1">
                {m.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
