import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiX, FiInfo } from 'react-icons/fi';
import { useCropPrediction } from '../hooks/useCropPrediction';
import PredictionResult from '../components/Prediction/PredictionResult';

const HomePage: React.FC = () => {
  const {
    imagePreview,
    prediction,
    isLoading,
    isReady,
    handleFileSelect,
    resetPrediction,
    topPredictions,
  } = useCropPrediction();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        handleFileSelect(acceptedFiles[0]);
      }
    },
    [handleFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    disabled: !isReady || isLoading,
  });

  if (!isReady) {
    return (
      <div className="text-center py-12">
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
        </div>
        <p className="mt-4 text-gray-600">Loading prediction service...</p>
        <p className="text-sm text-gray-500 mt-2">Please wait while we prepare the model.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Crop Health Prediction</h1>
        <p className="text-gray-600">Upload an image of a plant leaf to detect diseases and get health insights</p>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        {!imagePreview ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'
            }`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center justify-center">
              <div className="bg-primary-100 p-3 rounded-full mb-4">
                <FiUpload className="h-8 w-8 text-primary-600" />
              </div>
              {isDragActive ? (
                <p className="text-primary-600 font-medium">Drop the image here</p>
              ) : (
                <>
                  <p className="text-gray-700 font-medium">Drag & drop an image here, or click to select</p>
                  <p className="text-sm text-gray-500 mt-1">Supports JPG, PNG, WEBP (max 5MB)</p>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="relative">
            <div className="relative rounded-lg overflow-hidden bg-gray-100">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-full h-auto max-h-96 object-contain mx-auto"
              />
              <button
                onClick={resetPrediction}
                className="absolute top-2 right-2 bg-white rounded-full p-2 shadow-md hover:bg-gray-100 transition-colors"
                aria-label="Remove image"
              >
                <FiX className="h-5 w-5 text-gray-600" />
              </button>
            </div>

            <div className="mt-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Prediction Results</h3>
              {prediction ? (
                <PredictionResult prediction={prediction} topPredictions={topPredictions} />
              ) : isLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600 mx-auto"></div>
                  <p className="mt-4 text-gray-600">Analyzing image...</p>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <FiInfo className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                  <p>No prediction results available</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <div className="flex">
          <div className="flex-shrink-0">
            <FiInfo className="h-5 w-5 text-blue-400" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">Tips for best results</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ul className="list-disc pl-5 space-y-1">
                <li>Use clear, well-lit photos of individual leaves</li>
                <li>Ensure the leaf covers most of the image</li>
                <li>Avoid blurry or out-of-focus images</li>
                <li>Take photos against a contrasting background when possible</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
