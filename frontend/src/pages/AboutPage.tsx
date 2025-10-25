import React from 'react';
import { FiInfo, Framer, Code, Cpu, Shield, Users, Zap } from 'react-icons/fi';

const features = [
  {
    name: 'Advanced AI',
    description: 'Utilizes deep learning models to accurately identify plant diseases from leaf images.',
    icon: Cpu,
  },
  {
    name: 'Fast Predictions',
    description: 'Get instant results with our optimized prediction pipeline.',
    icon: Zap,
  },
  {
    name: 'User-Friendly',
    description: 'Simple and intuitive interface for users of all technical levels.',
    icon: Users,
  },
  {
    name: 'Secure',
    description: 'Your data privacy and security are our top priorities.',
    icon: Shield,
  },
];

const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">About Crop Health Prediction</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Empowering farmers and gardeners with AI-powered plant disease detection
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="prose max-w-none">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
          <p className="text-gray-700 mb-6">
            At Crop Health Prediction, we're on a mission to make plant disease detection accessible to everyone. 
            Our platform leverages cutting-edge machine learning to help farmers, gardeners, and agricultural 
            professionals quickly identify plant diseases and take appropriate action.
          </p>

          <h2 className="text-2xl font-bold text-gray-900 mb-4 mt-8">How It Works</h2>
          <ol className="list-decimal pl-5 space-y-4 text-gray-700">
            <li>
              <span className="font-medium">Upload an Image:</span> Take a clear photo of a plant leaf and upload it to our platform.
            </li>
            <li>
              <span className="font-medium">AI Analysis:</span> Our deep learning model analyzes the image to detect any signs of disease.
            </li>
            <li>
              <span className="font-medium">Get Results:</span> Receive instant, accurate predictions with confidence scores.
            </li>
            <li>
              <span className="font-medium">Take Action:</span> Use the provided information to take appropriate measures for plant health.
            </li>
          </ol>

          <h2 className="text-2xl font-bold text-gray-900 mb-4 mt-12">Features</h2>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {features.map((feature) => (
              <div key={feature.name} className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-50 text-primary-600">
                    <feature.icon className="h-6 w-6" />
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">{feature.name}</h3>
                  <p className="mt-1 text-gray-600">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>

          <h2 className="text-2xl font-bold text-gray-900 mb-4 mt-12">Technology Stack</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Frontend</h3>
                <ul className="space-y-2">
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    React with TypeScript
                  </li>
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    Tailwind CSS for styling
                  </li>
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    React Query for data fetching
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Backend</h3>
                <ul className="space-y-2">
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    FastAPI (Python)
                  </li>
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    TensorFlow/Keras for deep learning
                  </li>
                  <li className="flex items-center">
                    <span className="text-primary-600 mr-2">•</span>
                    Docker for containerization
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-12 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
            <div className="flex">
              <div className="flex-shrink-0">
                <FiInfo className="h-5 w-5 text-blue-400" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  <span className="font-medium">Note:</span> This application is for educational and research purposes. 
                  Always consult with agricultural experts for professional advice on plant health management.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
