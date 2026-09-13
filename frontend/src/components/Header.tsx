import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';
import footerLogo from '../assets/Quaz-logol.png';
import { HealthResponse } from '../types/eeg';

interface HeaderProps {
  health: HealthResponse | null;
  isHealthLoading: boolean;
  onNavigateToUpload?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, isHealthLoading, onNavigateToUpload }) => {
  return (
    <header className="sticky top-0 z-40 bg-[#243b53] border-b border-[#1b2d40] transition-all shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">

        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <a href="#" className="flex items-center">
            <img src={footerLogo} alt="Quantum EEG" className="w-20 h-20" />
            <div className="flex items-baseline gap-2">
              <span className="text-base font-bold tracking-tight text-white">
                Quantum EEG
              </span>
              <span className="text-[11px] font-mono font-medium text-slate-300 bg-[#1b2d40] px-1.5 py-0.5 rounded border border-[#334e68]">
                v1.3.0
              </span>
            </div>
          </a>
        </div>

        {/* Minimal Navigation */}
        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-300">
          <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
          <a href="#technology" className="hover:text-white transition-colors">Technology</a>
          <a href="#offline-evaluation" className="hover:text-white transition-colors">Evaluation</a>
        </nav>

        {/* Status & Action */}
        <div className="flex items-center gap-3">
          {/* Backend Status Pill */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-[#1b2d40] border border-[#334e68] text-slate-200">
            {isHealthLoading ? (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-slate-300">Connecting</span>
              </>
            ) : health ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span className="text-slate-200">VQC Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-rose-400" />
                <span className="text-rose-300">Offline</span>
              </>
            )}
          </div>

          {/* Analyze Button */}
          {onNavigateToUpload && (
            <button
              onClick={onNavigateToUpload}
              className="hidden sm:inline-flex items-center justify-center px-3.5 py-1.5 text-xs font-semibold text-white bg-sky-500 hover:bg-sky-400 rounded-lg shadow-sm transition-all hover:-translate-y-0.5 cursor-pointer"
            >
              Analyze EEG
            </button>
          )}
        </div>

      </div>
    </header>
  );
};
