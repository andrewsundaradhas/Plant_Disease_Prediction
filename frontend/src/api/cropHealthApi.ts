import axios, { AxiosError, AxiosResponse } from 'axios';
import { 
  PredictionResult, 
  ClassInfo, 
  HealthCheck, 
  ApiError, 
  EvaluationRequest, 
  EvaluationStatus, 
  EvaluationResults 
} from '../types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for adding auth token if needed
api.interceptors.request.use(
  (config) => {
    // You can add auth token here if needed
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for handling errors
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError<ApiError>) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      return Promise.reject({
        message: error.response.data?.message || 'An error occurred',
        status: error.response.status,
        details: error.response.data,
      });
    } else if (error.request) {
      // The request was made but no response was received
      return Promise.reject({
        message: 'No response from server. Please check your connection.',
      });
    } else {
      // Something happened in setting up the request that triggered an Error
      return Promise.reject({
        message: error.message || 'An error occurred',
      });
    }
  }
);

// Health check
export const healthCheck = async (): Promise<HealthCheck> => {
  const response = await api.get<HealthCheck>('/health');
  return response.data;
};

// Make prediction
export const predictImage = async (file: File): Promise<PredictionResult> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<PredictionResult>('/api/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Get available classes
export const getAvailableClasses = async (): Promise<ClassInfo> => {
  const response = await api.get<ClassInfo>('/api/classes');
  return response.data;
};

// Model Evaluation Endpoints
export const evaluateModel = async (request: EvaluationRequest): Promise<{ status: string }> => {
  const response = await api.post<{ status: string }>('/api/evaluate', request);
  return response.data;
};

export const getEvaluationStatus = async (): Promise<EvaluationStatus> => {
  const response = await api.get<EvaluationStatus>('/api/evaluation/status');
  return response.data;
};

export const getEvaluationResults = async (): Promise<EvaluationResults> => {
  const response = await api.get<EvaluationResults>('/api/evaluation/results');
  return response.data;
};

export const getConfusionMatrix = async (): Promise<Blob> => {
  const response = await api.get('/api/evaluation/confusion-matrix', {
    responseType: 'blob',
    params: { t: Date.now() } // Cache buster
  });
  return response.data;
};

export default api;
