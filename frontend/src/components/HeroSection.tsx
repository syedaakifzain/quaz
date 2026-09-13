import React from 'react';
import { ArrowRight, Sparkles, Cpu, Layers } from 'lucide-react';
import hero_atom from '../assets/hero_atom.png';

interface HeroSectionProps {
  onAnalyzeClick: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onAnalyzeClick }) => {
  return (
    <section className="relative pt-12 pb-20 md:pt-20 md:pb-32 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">

          {/* Left Column: Typography & CTAs */}
          <div className="lg:col-span-7 space-y-8">

            <h1 className="font-['Playfair_Display',_serif] text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.1]">
              Quantum <i>Electroencephalogram</i> Seizure Detection
            </h1>

            {/* Subheading */}
            <p className="text-lg sm:text-xl text-slate-600 leading-relaxed max-w-2xl font-normal">
              Quantum EEG Signal Classifier is a research prototype that processes EEG recordings, extracts signal features, and uses a 4-qubit Variational Quantum Classifier to estimate seizure-related activity across EEG segments.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4 pt-2">
              <button
                onClick={onAnalyzeClick}
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 text-base font-semibold text-white bg-sky-600 hover:bg-sky-700 rounded-xl shadow-sm hover:shadow transition-all hover:-translate-y-0.5 cursor-pointer"
              >
                Analyze an EEG
                <ArrowRight className="w-4 h-4" />
              </button>

              <a
                href="#technology"
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 text-base font-medium text-slate-700 hover:text-slate-900 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl shadow-subtle transition-all hover:-translate-y-0.5"
              >
                Explore the technology ↓
              </a>
            </div>

            {/* Trust highlights */}
            <div className="pt-6 border-t border-slate-200/70 flex items-center gap-6 text-xs text-slate-500 font-mono">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-sky-600" />
                4-Qubit VQC Engine
              </span>
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-sky-600" />
                20 Extracted Features
              </span>
            </div>

          </div>

          {/* Right Column: Quantum Brain Illustration */}
          <div className="lg:col-span-5 flex items-center justify-center lg:justify-end">
            <div className="relative w-full lg:w-[100%] lg:max-w-none lg:-mr-16">
              <img
                src={hero_atom}
                alt="Quantum computing and brain illustration"
                className="w-full h-auto object-contain drop-shadow-xl"
              />
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};
