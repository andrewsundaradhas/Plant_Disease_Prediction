import React from 'react';
import { PredictionResult } from '../../types';

interface PredictionResultProps {
  prediction: PredictionResult;
  topPredictions: Array<{ className: string; probability: number }>;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction, topPredictions }) => {
  const formatConfidence = (confidence: number) => {
    return (confidence * 100).toFixed(2) + '%';
  };

  const formatClassName = (name: string) => {
    // Convert snake_case to Title Case with spaces
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  return (
    <div className="space-y-6">
      {/* Main Prediction */}
      <div className="bg-green-50 p-4 rounded-lg border border-green-100">
        <div className="flex items-center">
          <div className="flex-shrink-0 h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
            <span className="text-green-600 text-xl font-bold">✓</span>
          </div>
          <div className="ml-4">
            <h4 className="text-lg font-medium text-gray-900">
              {formatClassName(prediction.predicted_class)}
            </h4>
            <div className="mt-1">
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-green-600 h-2.5 rounded-full"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-sm font-medium text-gray-700">
                  {formatConfidence(prediction.confidence)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Predictions */}
      {topPredictions.length > 1 && (
        <div>
          <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-3">
            Other Possible Matches
          </h4>
          <div className="space-y-3">
            {topPredictions.slice(1).map((item, index) => (
              <div key={index} className="flex items-center">
                <div className="w-24 text-sm text-gray-500">
                  {formatClassName(item.className)}
                </div>
                <div className="flex-1 ml-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${item.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div className="ml-3 w-16 text-right text-sm font-medium text-gray-700">
                  {formatConfidence(item.probability)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Information */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-3">
          Analysis Details
        </h4>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Model Confidence</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {formatConfidence(prediction.confidence)}
            </dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Timestamp</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {new Date(prediction.timestamp).toLocaleString()}
            </dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-sm font-medium text-gray-500">Image</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {prediction.image_path.split('/').pop()}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
};

export default PredictionResult;
