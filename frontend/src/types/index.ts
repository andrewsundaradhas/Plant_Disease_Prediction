export interface PredictionResult {
  image_path: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  timestamp: string;
}

export interface ClassInfo {
  count: number;
  classes: Record<number, string>;
}

export interface PredictionRequest {
  file: File;
}

export interface ApiError {
  message: string;
  status?: number;
  details?: any;
}

export interface HealthCheck {
  status: string;
  model_loaded: boolean;
}
