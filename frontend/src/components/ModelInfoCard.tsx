import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';
import { HealthResponse } from '../types/eeg';

interface ModelInfoCardProps {
  health: HealthResponse | null;
}

export const ModelInfoCard: React.FC<ModelInfoCardProps> = ({ health }) => {
  const specs = [
    { label: 'Classifier Type', value: health?.classifier || 'Variational Quantum Classifier (VQC)' },
    { label: 'Model Version', value: health?.model_version || 'v1.3.0' },
    { label: 'Quantum Qubits', value: `${health?.qubits || 4}  Qubits` },
    { label: 'Pre-PCA Feature Count', value: `${health?.features_before_pca || 20} Extracted Features` },
    { label: 'PCA Representation', value: `${health?.pca_components || 4} Orthogonal Components` },
    { label: 'Segment Window Duration', value: `${health?.segment_duration_seconds || 2} Seconds` },
    { label: 'Decision Threshold', value: `${(health?.threshold || 0.60).toFixed(2)} Probability` },
    { label: 'Training Subjects', value: 'CHB-MIT (chb01–chb05)' },
  ];

  return (
    <div className="surface-card rounded-2xl p-6 sm:p-8 shadow-premium">
      <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <Cpu className="w-5 h-5 text-sky-600" />
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            Model Specifications & Provenance
          </h3>
        </div>
        <span className="px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-sky-50 text-sky-700 border border-sky-200">
          Research Prototype · V1.3.0
        </span>
      </div>

      {/* Two-Column Clean Specification Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-xs font-mono">
        {specs.map((item, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0"
          >
            <span className="text-slate-500 font-sans">{item.label}</span>
            <span className="font-semibold text-slate-900">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
