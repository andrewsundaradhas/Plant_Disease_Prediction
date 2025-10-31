"""Comprehensive setup script for the Crop Health Prediction System."""
import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
import json
from typing import Optional, Dict, Any

# Configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ML_DIR = PROJECT_ROOT / "ml"
ENV_FILE = PROJECT_ROOT / ".env"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{'='*80}\n{text}\n{'='*80}{Colors.ENDC}\n")

def run_command(cmd: str, cwd: Optional[Path] = None, shell: bool = True) -> bool:
    """Run a shell command and return success status."""
    try:
        print(f"{Colors.OKBLUE}Running: {cmd}{Colors.ENDC}")
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            shell=shell,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}Error running command: {cmd}{Colors.ENDC}")
        print(f"{e.stderr}")
        return False

def check_python_version() -> bool:
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print(f"{Colors.FAIL}Error: Python 3.8 or higher is required{Colors.ENDC}")
        return False
    return True

def setup_virtualenv() -> bool:
    """Set up Python virtual environment."""
    venv_dir = PROJECT_ROOT / "venv"
    
    if not venv_dir.exists():
        print(f"{Colors.OKCYAN}Creating virtual environment...{Colors.ENDC}")
        if not run_command(f"{sys.executable} -m venv {venv_dir}"):
            return False
    
    # Activate virtual environment
    if platform.system() == "Windows":
        activate_script = venv_dir / "Scripts" / "activate.bat"
        pip_path = venv_dir / "Scripts" / "pip"
    else:
        activate_script = venv_dir / "bin" / "activate"
        pip_path = venv_dir / "bin" / "pip"
    
    os.environ["PATH"] = f"{venv_dir / 'Scripts' if platform.system() == 'Windows' else venv_dir / 'bin'}{os.pathsep}{os.environ['PATH']}"
    
    # Upgrade pip and setuptools
    if not run_command(f"{pip_path} install --upgrade pip setuptools wheel"):
        return False
    
    return True

def install_backend_dependencies() -> bool:
    """Install Python dependencies."""
    print(f"{Colors.OKCYAN}Installing backend dependencies...{Colors.ENDC}")
    
    # Install base requirements
    if not run_command("pip install -r requirements.txt", BACKEND_DIR):
        return False
    
    # Install ML requirements
    if not run_command("pip install -r requirements-ml.txt"):
        return False
    
    return True

def setup_database() -> bool:
    """Set up the database."""
    print(f"{Colors.OKCYAN}Setting up database...{Colors.ENDC}")
    
    # Create data directory if it doesn't exist
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Check if MongoDB is installed and running
    try:
        if platform.system() == "Windows":
            mongo_running = "MongoDB" in subprocess.check_output("sc query | findstr MongoDB", shell=True).decode()
        else:
            mongo_running = "mongod" in subprocess.check_output("ps aux | grep mongod", shell=True).decode()
            
        if not mongo_running:
            print(f"{Colors.WARNING}MongoDB is not running. Please start MongoDB service.{Colors.ENDC}")
            return False
            
    except subprocess.CalledProcessError:
        print(f"{Colors.WARNING}Could not check MongoDB status. Please ensure MongoDB is installed and running.{Colors.ENDC}")
        return False
    
    return True

def setup_frontend() -> bool:
    """Set up the frontend."""
    print(f"{Colors.OKCYAN}Setting up frontend...{Colors.ENDC}")
    
    # Create .env file if it doesn't exist
    env_file = FRONTEND_DIR / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("REACT_APP_API_URL=http://localhost:8000\n")
    
    # Install Node.js dependencies
    if not run_command("npm install", FRONTEND_DIR):
        return False
    
    return True

def setup_environment() -> bool:
    """Set up environment variables."""
    if not ENV_FILE.exists():
        print(f"{Colors.OKCYAN}Creating .env file...{Colors.ENDC}")
        env_config = {
            "DEBUG": "True",
            "MONGODB_URI": "mongodb://localhost:27017/crop_health",
            "SECRET_KEY": "your-secret-key-change-this-in-production",
            "AWS_ACCESS_KEY_ID": "your-aws-access-key",
            "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key",
            "AWS_REGION": "us-east-1",
            "AWS_S3_BUCKET": "crop-health-data"
        }
        
        with open(ENV_FILE, 'w') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
        
        print(f"{Colors.WARNING}Please edit {ENV_FILE} with your configuration.{Colors.ENDC}")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
    
    return True

def verify_setup() -> bool:
    """Verify the setup was successful."""
    print(f"{Colors.OKCYAN}Verifying setup...{Colors.ENDC}")
    
    # Check Python packages
    try:
        import fastapi
        import tensorflow
        import pymongo
        import uvicorn
        import numpy
        import pandas
        import sklearn
    except ImportError as e:
        print(f"{Colors.FAIL}Missing Python package: {e.name}{Colors.ENDC}")
        return False
    
    # Check Node.js and npm
    try:
        node_version = subprocess.check_output("node --version", shell=True).decode().strip()
        npm_version = subprocess.check_output("npm --version", shell=True).decode().strip()
        print(f"Node.js version: {node_version}")
        print(f"npm version: {npm_version}")
    except subprocess.CalledProcessError:
        print(f"{Colors.FAIL}Node.js and npm are required for the frontend{Colors.ENDC}")
        return False
    
    return True

def main() -> None:
    """Main setup function."""
    print_header("Crop Health Prediction System - Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up virtual environment
    if not setup_virtualenv():
        sys.exit(1)
    
    # Set up environment variables
    if not setup_environment():
        sys.exit(1)
    
    # Install backend dependencies
    if not install_backend_dependencies():
        sys.exit(1)
    
    # Set up database
    if not setup_database():
        print(f"{Colors.WARNING}Database setup may require manual intervention.{Colors.ENDC}")
    
    # Set up frontend
    if not setup_frontend():
        print(f"{Colors.WARNING}Frontend setup may require manual intervention.{Colors.ENDC}")
    
    # Verify setup
    if not verify_setup():
        print(f"{Colors.WARNING}Some setup steps may require manual intervention.{Colors.ENDC}")
    
    print_header("Setup Complete!")
    print(f"{Colors.OKGREEN}✅ Setup completed successfully!{Colors.ENDC}")
    print("\nNext steps:")
    print(f"1. Edit {ENV_FILE} with your configuration")
    print("2. Start the backend server:")
    print(f"   cd {BACKEND_DIR}")
    print("   uvicorn app.main:app --reload")
    print("3. Start the frontend development server:")
    print(f"   cd {FRONTEND_DIR}")
    print("   npm start")
    print("\nAccess the application at http://localhost:3000")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}An error occurred during setup:{Colors.ENDC}")
        print(f"{type(e).__name__}: {str(e)}")
        sys.exit(1)
