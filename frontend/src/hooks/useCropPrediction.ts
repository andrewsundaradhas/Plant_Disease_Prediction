import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { predictImage, getAvailableClasses, healthCheck, PredictionResult, ClassInfo, HealthCheck } from '../api/cropHealthApi';
import { toast } from 'react-hot-toast';

export const useCropPrediction = () => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  // Health check query
  const healthQuery = useQuery<HealthCheck, Error>(
    ['health'],
    healthCheck,
    {
      refetchOnWindowFocus: false,
      onError: (error) => {
        console.error('Health check failed:', error);
        toast.error('Failed to connect to the prediction service');
      },
    }
  );

  // Available classes query
  const classesQuery = useQuery<ClassInfo, Error>(
    ['classes'],
    getAvailableClasses,
    {
      enabled: healthQuery.data?.status === 'ok',
      onError: (error) => {
        console.error('Failed to fetch classes:', error);
        toast.error('Failed to load available crop classes');
      },
    }
  );

  // Prediction mutation
  const predictMutation = useMutation<PredictionResult, Error, File>(
    (file) => predictImage(file),
    {
      onSuccess: (data) => {
        setPrediction(data);
        toast.success('Prediction completed successfully');
      },
      onError: (error: Error) => {
        console.error('Prediction failed:', error);
        toast.error(`Prediction failed: ${error.message}`);
        setPrediction(null);
      },
    }
  );

  // Handle file selection
  const handleFileSelect = (file: File | null) => {
    if (!file) {
      setImagePreview(null);
      setPrediction(null);
      return;
    }

    // Check file type
    if (!file.type.match('image.*')) {
      toast.error('Please select an image file');
      return;
    }

    // Check file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      toast.error('Image size should be less than 5MB');
      return;
    }

    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Reset previous prediction
    setPrediction(null);

    // Make prediction
    predictMutation.mutate(file);
  };

  // Get top predictions (sorted by probability)
  const getTopPredictions = (count: number = 3) => {
    if (!prediction) return [];
    
    return Object.entries(prediction.class_probabilities)
      .sort(([, a], [, b]) => b - a)
      .slice(0, count)
      .map(([className, probability]) => ({
        className,
        probability,
      }));
  };

  // Check if the service is healthy and ready
  const isReady = healthQuery.data?.status === 'ok' && healthQuery.data.model_loaded;

  return {
    // State
    prediction,
    imagePreview,
    isReady,
    isLoading: predictMutation.isLoading,
    isError: predictMutation.isError,
    error: predictMutation.error,
    availableClasses: classesQuery.data,
    topPredictions: getTopPredictions(3),
    
    // Actions
    handleFileSelect,
    resetPrediction: () => {
      setPrediction(null);
      setImagePreview(null);
    },
    
    // Queries
    healthQuery,
    classesQuery,
    predictMutation,
  };
};

export default useCropPrediction;
