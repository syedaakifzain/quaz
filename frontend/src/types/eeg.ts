export interface SegmentPrediction {
  segment_index: number;
  start_time_seconds: number;
  end_time_seconds: number;
  predicted_label: number; // 0 = Non-Seizure, 1 = Seizure
  label_name: string; // "Non-Seizure" | "Seizure"
  seizure_probability: number;
}

export interface AnalysisResult {
  recording_name: string;
  duration_seconds: number;
  sampling_frequency: number;
  channels_used: string[];
  total_segments: number;
  seizure_segments: number;
  non_seizure_segments: number;
  prediction_summary: string;
  segment_predictions: SegmentPrediction[];
  model_version: string;
  processing_time_seconds: number;
  warnings?: string[];
  timestamp?: string;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  classifier: string;
  threshold: number;
  qubits: number;
  features_before_pca: number;
  pca_components: number;
  channels: string[];
  segment_duration_seconds: number;
  sampling_frequency: number;
}

export type AppState = 'IDLE' | 'UPLOADING' | 'PROCESSING' | 'RESULTS' | 'ERROR';
