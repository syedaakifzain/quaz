import React from 'react';
import { Cpu, ArrowRight, Zap, Database, GitMerge, CheckCircle2 } from 'lucide-react';

export const TechnologySection: React.FC = () => {
  const circuitLayers = [
    { name: 'q[0]', label: 'PCA Dim 0' },
    { name: 'q[1]', label: 'PCA Dim 1' },
    { name: 'q[2]', label: 'PCA Dim 2' },
    { name: 'q[3]', label: 'PCA Dim 3' },
  ];

  return (
    <section id="technology" className="py-24 sm:py-32 bg-slate-100/60 border-y border-slate-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center">

          {/* Left Column: Explanation */}
          <div className="lg:col-span-6 space-y-6">
            <p className="text-xs font-mono font-semibold tracking-wider text-sky-600 uppercase">
              Quantum Architecture
            </p>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
              Built around a quantum machine learning pipeline.
            </h2>
            <p className="text-base text-slate-600 leading-relaxed">
              The quantum circuit maps the four PCA components into a parameterized quantum representation for classification.
            </p>

            {/* Architecture sequence chain */}
            <div className="pt-2 space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                <span className="px-2.5 py-1 rounded bg-white border border-slate-200 text-slate-700 font-medium">StandardScaler</span>
                <span className="text-slate-400">→</span>
                <span className="px-2.5 py-1 rounded bg-white border border-slate-200 text-slate-700 font-medium">PCA (4D)</span>
                <span className="text-slate-400">→</span>
                <span className="px-2.5 py-1 rounded bg-white border border-slate-200 text-slate-700 font-medium">MinMaxScaler [0, π]</span>
                <span className="text-slate-400">→</span>
                <span className="px-2.5 py-1 rounded bg-sky-50 border border-sky-200 text-sky-700 font-medium">ZZFeatureMap</span>
                <span className="text-slate-400">→</span>
                <span className="px-2.5 py-1 rounded bg-sky-50 border border-sky-200 text-sky-700 font-medium">RealAmplitudes</span>
                <span className="text-slate-400">→</span>
                <span className="px-2.5 py-1 rounded bg-slate-900 text-white font-medium">VQC Engine</span>
              </div>
            </div>

            {/* Technical bullets */}
            <div className="pt-4 space-y-2.5 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-sky-600 shrink-0" />
                <span><strong>ZZFeatureMap:</strong> 2 repetitions with full entanglement for cross-feature correlation.</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-sky-600 shrink-0" />
                <span><strong>RealAmplitudes Ansatz:</strong> Parameterized rotation gates (Ry) & circular CNOT entanglement.</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-sky-600 shrink-0" />
                <span>
                  <strong>VQC Inference:</strong> The trained quantum classifier generates
                  a probability score for each EEG segment.
                </span>
              </div>
            </div>

          </div>

          {/* Right Column: Quantum Circuit Visualizer */}
          <div className="lg:col-span-6">
            <div className="surface-card rounded-2xl p-6 sm:p-8 shadow-premium">
              <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-sky-600" />
                  <h3 className="text-sm font-semibold text-slate-900">4-Qubit Variational Quantum Circuit</h3>
                </div>
                <span className="text-xs font-mono text-slate-500">v1.3.0 Architecture</span>
              </div>

              {/* Minimalist Circuit Diagram */}
              <div className="space-y-4 font-mono text-xs">
                {circuitLayers.map((wire, wIdx) => (
                  <div key={wIdx} className="flex items-center gap-3">
                    <span className="w-10 text-slate-500 font-semibold">{wire.name}</span>
                    <div className="flex-1 flex items-center gap-2 py-2 px-3 bg-slate-50 rounded-lg border border-slate-200/80 overflow-x-auto">
                      {/* State prep */}
                      <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 shrink-0 text-[11px]">|0⟩</span>
                      <span className="text-slate-300">──</span>
                      {/* Feature map gate */}
                      <span className="px-2 py-0.5 rounded bg-sky-100/80 border border-sky-200 text-sky-800 font-medium shrink-0 text-[11px]">
                        U_Φ(x_{wIdx})
                      </span>
                      <span className="text-slate-300">──</span>
                      {/* CNOT entangler visual */}
                      <span className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-700 text-[10px] shrink-0">
                        ●
                      </span>
                      <span className="text-slate-300">──</span>
                      {/* Variational gate */}
                      <span className="px-2 py-0.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-700 font-medium shrink-0 text-[11px]">
                        Ry(θ_{wIdx})
                      </span>
                      <span className="text-slate-300">──</span>
                      {/* Measurement */}
                      <span className="px-2 py-0.5 rounded bg-slate-900 text-white shrink-0 text-[11px]">
                        ⟨Z⟩
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>4 Qubits</span>
                <span>Ansatz Reps: 2</span>
                <span>Trainable Parameters: 12</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};
