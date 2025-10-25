import React from 'react';
import { Link } from 'react-router-dom';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow">
      <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
        <Link to="/" className="flex items-center">
          <div className="flex-shrink-0">
            <img
              className="h-8 w-auto"
              src="/logo.svg"
              alt="Crop Health"
            />
          </div>
          <h1 className="ml-3 text-xl font-bold text-gray-900">Crop Health Prediction</h1>
        </Link>
        <nav className="hidden md:flex space-x-8">
          <Link to="/" className="text-gray-900 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
            Home
          </Link>
          <Link to="/about" className="text-gray-500 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
            About
          </Link>
          <Link to="/dashboard" className="text-gray-500 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
            Dashboard
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
