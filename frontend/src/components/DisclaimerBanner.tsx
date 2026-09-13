import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div id="disclaimer" className="p-4 rounded-xl bg-slate-100/70 border border-slate-200 text-slate-600 text-xs flex items-start gap-3">
      <ShieldAlert className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
      <p className="leading-relaxed">
        <strong className="text-slate-800 font-semibold">Research Prototype Notice:</strong> This system provides experimental AI-assisted EEG signal analysis based on Variational Quantum Classification. It is intended strictly for academic research and algorithm benchmarking and must not be used as a standalone diagnostic tool or for clinical treatment decisions.
      </p>
    </div>
  );
};
