import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { evaluateModel, getEvaluationStatus, getEvaluationResults } from '../api/cropHealthApi';
import { toast } from 'react-hot-toast';
import { FiRefreshCw, FiAlertCircle, FiCheckCircle, FiClock } from 'react-icons/fi';

const ModelEvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const [testDataDir, setTestDataDir] = useState('');
  const [batchSize, setBatchSize] = useState(32);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Query for evaluation status
  const { data: statusData, refetch: refetchStatus } = useQuery(
    ['evaluationStatus'],
    getEvaluationStatus,
    {
      refetchInterval: 5000, // Poll every 5 seconds
      enabled: isEvaluating,
      onSuccess: (data) => {
        if (!data.evaluation_in_progress && isEvaluating) {
          setIsEvaluating(false);
          if (data.results_available) {
            refetchResults();
            toast.success('Model evaluation completed!');
          }
        }
      },
    }
  );

  // Query for evaluation results
  const { 
    data: results, 
    isLoading: isLoadingResults, 
    refetch: refetchResults 
  } = useQuery(
    ['evaluationResults'],
    getEvaluationResults,
    {
      enabled: false, // Don't fetch on mount
      onError: (error: any) => {
        toast.error(error.message || 'Failed to load evaluation results');
      },
    }
  );

  // Mutation for starting evaluation
  const startEvaluation = useMutation(
    () => evaluateModel({ test_data_dir: testDataDir, batch_size: batchSize }),
    {
      onSuccess: () => {
        setIsEvaluating(true);
        toast.success('Model evaluation started. This may take a few minutes...');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to start evaluation');
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!testDataDir.trim()) {
      toast.error('Please enter the test data directory');
      return;
    }
    startEvaluation.mutate();
  };

  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(2) + '%';
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Model Evaluation</h1>
      
      <div className="bg-white shadow rounded-lg p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Run Evaluation</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="testDataDir" className="block text-sm font-medium text-gray-700 mb-1">
              Test Data Directory
            </label>
            <input
              type="text"
              id="testDataDir"
              value={testDataDir}
              onChange={(e) => setTestDataDir(e.target.value)}
              placeholder="e.g., /path/to/test/data"
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              disabled={isEvaluating}
            />
            <p className="mt-1 text-sm text-gray-500">
              Path to the directory containing test images organized by class
            </p>
          </div>
          
          <div className="w-1/3">
            <label htmlFor="batchSize" className="block text-sm font-medium text-gray-700 mb-1">
              Batch Size
            </label>
            <input
              type="number"
              id="batchSize"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)}
              min={1}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              disabled={isEvaluating}
            />
          </div>
          
          <div className="pt-2">
            <button
              type="submit"
              disabled={isEvaluating || startEvaluation.isLoading}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                isEvaluating || startEvaluation.isLoading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500'
              }`}
            >
              {isEvaluating || startEvaluation.isLoading ? (
                <>
                  <FiRefreshCw className="animate-spin mr-2" />
                  Evaluating...
                </>
              ) : (
                'Start Evaluation'
              )}
            </button>
          </div>
        </form>
        
        {isEvaluating && (
          <div className="mt-6 p-4 bg-blue-50 rounded-md border border-blue-200">
            <div className="flex items-center">
              <FiRefreshCw className="animate-spin text-blue-500 mr-2" />
              <span className="text-blue-700">Evaluation in progress. This may take several minutes...</span>
            </div>
            <div className="mt-2 text-sm text-blue-600">
              <p>Status: {statusData?.evaluation_in_progress ? 'Running' : 'Processing...'}</p>
            </div>
          </div>
        )}
      </div>
      
      {results && (
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold">Evaluation Results</h2>
            <p className="text-sm text-gray-500">
              Last updated: {new Date(results.timestamp).toLocaleString()}
            </p>
          </div>
          
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard 
                title="Accuracy" 
                value={formatPercentage(results.accuracy)} 
                icon={<FiCheckCircle className="text-green-500" />}
              />
              <MetricCard 
                title="Precision" 
                value={formatPercentage(results.precision)} 
                icon={<FiCheckCircle className="text-blue-500" />}
              />
              <MetricCard 
                title="Recall" 
                value={formatPercentage(results.recall)} 
                icon={<FiCheckCircle className="text-purple-500" />}
              />
              <MetricCard 
                title="F1 Score" 
                value={formatPercentage(results.f1_score)} 
                icon={<FiCheckCircle className="text-yellow-500" />}
              />
            </div>
            
            <div className="mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Confusion Matrix</h3>
              {results.confusion_matrix_plot ? (
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <img 
                    src={`/api/evaluation/confusion-matrix?t=${new Date().getTime()}`} 
                    alt="Confusion Matrix" 
                    className="max-w-full h-auto mx-auto"
                  />
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No confusion matrix available. Run an evaluation to generate one.
                </div>
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Class-wise Metrics</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Class</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precision</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recall</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1-Score</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Support</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {Object.entries(results.class_metrics).map(([className, metrics]: [string, any]) => (
                      <tr key={className}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {className}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatPercentage(metrics.precision)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatPercentage(metrics.recall)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatPercentage(metrics.f1_score)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.support}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon }) => (
  <div className="bg-white p-6 rounded-lg shadow border border-gray-100">
    <div className="flex items-center">
      <div className="p-3 rounded-full bg-opacity-10 bg-gray-100 mr-4">
        {icon}
      </div>
      <div>
        <p className="text-sm font-medium text-gray-500">{title}</p>
        <p className="text-2xl font-semibold text-gray-900">{value}</p>
      </div>
    </div>
  </div>
);

export default ModelEvaluationPage;
