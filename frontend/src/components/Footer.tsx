import React from 'react';
import footerLogo from '../assets/quaz-logo.png';
import footerBg from '../assets/footer-bg.png';

export const Footer: React.FC = () => {
  return (
    <footer className="relative border-t border-stone-300/80 bg-[#f7f2de] text-slate-900 text-sm overflow-hidden">
      {/* Background illustration */}
      <div
        className="absolute inset-0 bg-no-repeat bg-bottom bg-cover opacity-50 pointer-events-none"
        style={{ backgroundImage: `url(${footerBg})` }}
      />

      {/* Clean content directly over background */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 lg:gap-12">

          {/* Brand & Mission */}
          <div className="md:col-span-5 space-y-3">
            <div className="flex items-center">
              <img src={footerLogo} alt="Quaz-logo" className="w-14 h-14 object-contain" />
              <span className="text-base font-bold text-slate-900 tracking-tight">
                Quantum EEG
              </span>
            </div>
            <p className="text-slate-800 text-sm max-w-sm leading-relaxed">
              AI-assisted EEG signal analysis using a variational quantum classifier. The pipeline transforms multi-channel EEG recordings into engineered features and quantum-model probability estimates for seizure classification.</p>
            <p className="text-xs text-slate-700 font-mono">
              Research prototype • Powered by Qiskit & PyWavelets
            </p>
          </div>

          {/* Product Links */}
          <div className="md:col-span-3 space-y-3">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-mono">
              Platform
            </h4>
            <ul className="space-y-2 text-sm text-slate-800 font-medium">
              <li>
                <a href="#analyze" className="hover:text-sky-900 hover:underline transition-colors">Analyze EEG</a>
              </li>
              <li>
                <a href="#how-it-works" className="hover:text-sky-900 hover:underline transition-colors">Pipeline</a>
              </li>
              <li>
                <a href="#technology" className="hover:text-sky-900 hover:underline transition-colors">Quantum Technology</a>
              </li>
              <li>
                <a href="#offline-evaluation" className="hover:text-sky-900 hover:underline transition-colors">Research Benchmarks</a>
              </li>
            </ul>
          </div>

          {/* Project & Tech Specs */}
          <div className="md:col-span-4 space-y-3">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider font-mono">
              Project Specification
            </h4>
            <ul className="space-y-2 text-sm text-slate-800">
              <li className="flex justify-between">
                <span><b>Model Version:</b></span>
                <span className="font-mono font-bold text-slate-900">VQC v1.3.0</span>
              </li>

              <li className="flex justify-between">
                <span><b>Quantum Ansatz:</b></span>
                <span className="font-mono font-bold text-slate-900">RealAmplitude (4Q)</span>
              </li>

              <li className="flex justify-between">
                <span><b>Feature Extraction:</b></span>
                <span className="font-mono font-bold text-slate-900">
                  Wavelet + Statistical Features (20-Dim)
                </span>
              </li>

              <li className="flex justify-between">
                <span><b> Decision Threshold:</b></span>
                <span className="font-mono font-bold text-slate-900">0.60 Probability</span>
              </li>
            </ul>
          </div>

        </div>

        {/* Bottom Bar */}
        <div className="mt-14 pt-6 border-t border-stone-400/40 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-medium text-slate-700">
          <p>
            © 2026 Quantum EEG Project. Research prototype. Not intended for clinical diagnosis.
          </p>
          <div className="flex items-center gap-4 text-slate-700 font-mono">
            <span>CHB-MIT Benchmark</span>
            <span>•</span>
            <span>Qiskit Quantum Simulator</span>
          </div>
        </div>

      </div>
    </footer>
  );
};
