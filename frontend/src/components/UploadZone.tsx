import React, { useState, useRef } from 'react';
import { Upload, FileCode, Play, AlertCircle, FileCheck, ArrowRight, ShieldCheck } from 'lucide-react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  onDemoSelect: () => void;
  isBackendAvailable: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileSelect,
  onDemoSelect,
  isBackendAvailable,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File | null) => {
    if (!file) return;
    setValidationError(null);

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'edf') {
      setValidationError(`Invalid format (.${ext}). Please upload an EDF (.edf) recording.`);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <section id="analyze" className="py-24 sm:py-32">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-12">
          <p className="text-xs font-mono font-semibold tracking-wider text-sky-600 uppercase mb-2">
            EEG INFERENCE
          </p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Analyze an EEG <i>Recording</i>.
          </h2>
          <p className="text-base text-slate-600 mt-2">
            Upload an EDF recording and let the <i className='text-sky-600 font-bold' >V1.3.0</i> pipeline analyze it for seizure-like activity and generate interpretable probability results.
          </p>
        </div>

        {/* Validation Error Notice */}
        {validationError && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-center gap-3 text-rose-800 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {/* Minimal Soft Rounded Dropzone */}
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative rounded-2xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-200 border-2 border-dashed ${dragActive
            ? 'border-sky-500 bg-sky-50/60 scale-[1.005]'
            : selectedFile
              ? 'border-sky-400 bg-sky-50/30'
              : 'border-stone-300 hover:border-stone-400 bg-[#FAF7ED] hover:bg-[#f3eedd] shadow-subtle'
            }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".edf"
            onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
            className="hidden"
          />

          {selectedFile ? (
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-sky-100 flex items-center justify-center text-sky-600 mb-3">
                <FileCheck className="w-6 h-6" />
              </div>
              <p className="text-base font-semibold text-slate-900 font-mono">
                {selectedFile.name}
              </p>
              <p className="text-xs text-slate-500 font-mono mt-1">
                {formatFileSize(selectedFile.size)} • Standard EDF Recording
              </p>
              <span className="mt-4 text-xs font-medium text-sky-600 hover:text-sky-700 underline">
                Choose a different file
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-stone-200/70 flex items-center justify-center text-slate-700 mb-4 group-hover:scale-105 transition-transform">
                <Upload className="w-6 h-6 text-slate-700" />
              </div>
              <p className="text-base font-semibold text-slate-900">
                Drop your EDF recording here
                <br />
                or browse files from your computer
              </p>
              <p className="text-sm text-slate-500 mt-1">
                or <span className="text-sky-600 font-medium hover:underline">browse files</span> from your computer
              </p>
              <span className="mt-4 text-xs text-slate-400 font-mono">
                EDF files only (European Data Format)
              </span>
            </div>
          )}
        </div>

        {/* Action Controls */}
        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            disabled={!selectedFile || !isBackendAvailable}
            onClick={() => selectedFile && onFileSelect(selectedFile)}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 shadow-sm transition-all ${selectedFile && isBackendAvailable
              ? 'bg-slate-900 hover:bg-slate-800 text-white cursor-pointer hover:-translate-y-0.5 shadow'
              : 'bg-slate-200 text-slate-400 cursor-not-allowed'
              }`}
          >
            <Play className="w-4 h-4 fill-current" />
            Analyze recording
          </button>

          <span className="text-xs text-slate-400 font-mono uppercase">or</span>

          <button
            disabled={!isBackendAvailable}
            onClick={onDemoSelect}
            className={`w-full sm:w-auto px-6 py-3.5 rounded-xl font-medium text-sm border flex items-center justify-center gap-2 transition-all ${isBackendAvailable
              ? 'border-stone-300 bg-[#faf7ee] hover:bg-[#f3eedd] text-slate-700 hover:text-slate-900 cursor-pointer hover:-translate-y-0.5 shadow-subtle'
              : 'border-slate-200 bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
          >
            <FileCode className="w-4 h-4 text-sky-600" />
            Try demo recording (chb06_01.edf)
          </button>
        </div>

        {/* Backend offline warning */}
        {!isBackendAvailable && (
          <p className="mt-4 text-center text-xs text-rose-600 font-mono">
            Backend service offline. Please start FastAPI on http://localhost:8000
          </p>
        )}

      </div>
    </section>
  );
};
