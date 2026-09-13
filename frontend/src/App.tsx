import React, { useEffect, useState } from 'react';
import { AppState, AnalysisResult, HealthResponse, SegmentPrediction } from './types/eeg';
import { getHealth, analyzeEDF, analyzeDemo } from './services/api';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { MetricsStrip } from './components/MetricsStrip';
import { HowItWorks } from './components/HowItWorks';
import { TechnologySection } from './components/TechnologySection';
import { UploadZone } from './components/UploadZone';
import { OfflineEvaluation } from './components/OfflineEvaluation';
import { ProcessingOverlay } from './components/ProcessingOverlay';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { SummaryCards } from './components/SummaryCards';
import { DetectionStatusBanner } from './components/DetectionStatusBanner';
import { ProbabilityChart } from './components/ProbabilityChart';
import { DetectionTimeline } from './components/DetectionTimeline';
import { DetectionTable } from './components/DetectionTable';
import { SelectedSegment } from './components/SelectedSegment';
import { ModelInfoCard } from './components/ModelInfoCard';
import { ExportMenu } from './components/ExportMenu';
import { Footer } from './components/Footer';
import { AlertCircle, ArrowLeft, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthLoading, setIsHealthLoading] = useState<boolean>(true);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<SegmentPrediction | null>(null);

  // Fetch backend health status on mount
  useEffect(() => {
    let isMounted = true;
    const fetchHealthStatus = async () => {
      try {
        setIsHealthLoading(true);
        const data = await getHealth();
        if (isMounted) {
          setHealth(data);
          setIsHealthLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setHealth(null);
          setIsHealthLoading(false);
        }
      }
    };

    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Handle uploaded EDF analysis
  const handleAnalyzeFile = async (file: File) => {
    try {
      setAppState('PROCESSING');
      setErrorMessage(null);
      const result = await analyzeEDF(file);
      setAnalysisResult(result);
      if (result.segment_predictions && result.segment_predictions.length > 0) {
        const firstFlagged = result.segment_predictions.find(
          (s) => s.seizure_probability >= (health?.threshold || 0.60)
        );
        setSelectedSegment(firstFlagged || result.segment_predictions[0]);
      }
      setAppState('RESULTS');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred while executing the inference pipeline.');
      setAppState('ERROR');
    }
  };

  // Handle demo analysis
  const handleAnalyzeDemo = async () => {
    try {
      setAppState('PROCESSING');
      setErrorMessage(null);
      const result = await analyzeDemo();
      setAnalysisResult(result);
      if (result.segment_predictions && result.segment_predictions.length > 0) {
        const firstFlagged = result.segment_predictions.find(
          (s) => s.seizure_probability >= (health?.threshold || 0.60)
        );
        setSelectedSegment(firstFlagged || result.segment_predictions[0]);
      }
      setAppState('RESULTS');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to execute demo inference pipeline.');
      setAppState('ERROR');
    }
  };

  // Reset App State to IDLE
  const handleReset = () => {
    setAppState('IDLE');
    setAnalysisResult(null);
    setErrorMessage(null);
    setSelectedSegment(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToUpload = () => {
    if (appState !== 'IDLE') {
      handleReset();
    } else {
      const el = document.getElementById('analyze');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const threshold = health?.threshold || 0.60;

  return (
    <div className="min-h-screen text-slate-900 flex flex-col font-sans selection:bg-sky-500 selection:text-white">

      {/* Sticky Header */}
      <Header
        health={health}
        isHealthLoading={isHealthLoading}
        onNavigateToUpload={scrollToUpload}
      />

      {/* Main Container */}
      <main className="flex-1">

        {/* State 1: IDLE — Full Landing Page Experience */}
        {appState === 'IDLE' && (
          <div>
            <HeroSection onAnalyzeClick={scrollToUpload} />
            <MetricsStrip />
            <HowItWorks />
            <TechnologySection />
            <UploadZone
              onFileSelect={handleAnalyzeFile}
              onDemoSelect={handleAnalyzeDemo}
              isBackendAvailable={!!health}
            />
            <OfflineEvaluation />
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
              <DisclaimerBanner />
            </div>
          </div>
        )}

        {/* State 2: PROCESSING — Focused Loading Experience */}
        {(appState === 'UPLOADING' || appState === 'PROCESSING') && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <ProcessingOverlay />
          </div>
        )}

        {/* State 3: ERROR — Clean Alert Card */}
        {appState === 'ERROR' && (
          <div className="max-w-xl mx-auto px-4 py-24">
            <div className="surface-card p-8 rounded-2xl text-center flex flex-col items-center shadow-premium">
              <div className="w-12 h-12 rounded-full bg-rose-100 text-rose-600 flex items-center justify-center mb-4">
                <AlertCircle className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">Analysis Failed</h3>
              <p className="text-xs text-slate-600 font-mono mt-3 bg-slate-50 p-4 rounded-xl border border-slate-200 max-w-md break-words">
                {errorMessage}
              </p>
              <button
                onClick={handleReset}
                className="mt-6 px-6 py-2.5 rounded-xl font-medium text-xs bg-slate-900 hover:bg-slate-800 text-white flex items-center gap-2 transition-all hover:-translate-y-0.5 cursor-pointer"
              >
                <RefreshCw className="w-4 h-4" />
                Return to Home
              </button>
            </div>
          </div>
        )}

        {/* State 4: RESULTS — Comprehensive Analysis Dashboard */}
        {appState === 'RESULTS' && analysisResult && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8 animate-fade-in">

            {/* Top Navigation & Breadcrumbs */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-200/80">
              <button
                onClick={handleReset}
                className="inline-flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-slate-900 transition-colors cursor-pointer group"
              >
                <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
                Back to recordings
              </button>

              <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
                <span>Recording:</span>
                <span className="font-semibold text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                  {analysisResult.recording_name}
                </span>
              </div>
            </div>

            {/* High-Impact Detection Status Banner */}
            <DetectionStatusBanner result={analysisResult} threshold={threshold} />

            {/* Typography-led Summary Metrics */}
            <SummaryCards result={analysisResult} threshold={threshold} />

            {/* Probability Graph */}
            <ProbabilityChart
              segmentPredictions={analysisResult.segment_predictions}
              threshold={threshold}
              selectedSegmentIndex={selectedSegment?.segment_index ?? null}
              onSelectSegment={(seg) => setSelectedSegment(seg)}
            />

            {/* Continuous Timeline */}
            <DetectionTimeline
              segmentPredictions={analysisResult.segment_predictions}
              threshold={threshold}
              selectedSegmentIndex={selectedSegment?.segment_index ?? null}
              onSelectSegment={(seg) => setSelectedSegment(seg)}
            />

            {/* Selected Segment Inspection */}
            <SelectedSegment segment={selectedSegment} threshold={threshold} />

            {/* Detection Table */}
            <DetectionTable
              segmentPredictions={analysisResult.segment_predictions}
              threshold={threshold}
              selectedSegmentIndex={selectedSegment?.segment_index ?? null}
              onSelectSegment={(seg) => setSelectedSegment(seg)}
            />

            {/* Model Details & Provenance */}
            <ModelInfoCard health={health} />

            {/* Offline Model Evaluation Reference */}
            <OfflineEvaluation />

            {/* Export & Reset Actions */}
            <ExportMenu result={analysisResult} onReset={handleReset} />

            {/* Disclaimer */}
            <DisclaimerBanner />

          </div>
        )}

      </main>

      {/* Global Minimal Footer */}
      <Footer />

    </div>
  );
};
