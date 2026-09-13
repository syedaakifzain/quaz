import { AnalysisResult, HealthResponse } from '../types/eeg';

const API_BASE_URL = '/api';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      const errText = await response.text();
      throw new ApiError(response.status, `Backend health check failed: ${errText}`);
    }
    return (await response.json()) as HealthResponse;
  } catch (error: any) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, `Backend service unavailable (${error.message || 'Network error'})`);
  }
}

export async function analyzeEDF(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMsg = `Server error (${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        errorMsg = await response.text();
      }
      throw new ApiError(response.status, errorMsg);
    }

    return (await response.json()) as AnalysisResult;
  } catch (error: any) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, `Analysis failed: ${error.message || 'Network error'}`);
  }
}

export async function analyzeDemo(): Promise<AnalysisResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze-demo`, {
      method: 'POST',
    });

    if (!response.ok) {
      let errorMsg = `Demo analysis failed (${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        errorMsg = await response.text();
      }
      throw new ApiError(response.status, errorMsg);
    }

    return (await response.json()) as AnalysisResult;
  } catch (error: any) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, `Demo analysis request failed: ${error.message || 'Network error'}`);
  }
}
