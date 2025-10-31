"""
Setup script for new machines.
Run this after cloning the repository to set up the environment.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and print output in real-time."""
    print(f"\nRunning: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        return process.wait() == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("\n=== Setting up Python environment ===")
    
    # Create virtual environment
    venv_name = "venv"
    if not os.path.exists(venv_name):
        print(f"Creating virtual environment: {venv_name}")
        if not run_command(f"python -m venv {venv_name}"):
            return False
    
    # Activate virtual environment and install requirements
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_name, "Scripts", "activate")
        pip_cmd = os.path.join(venv_name, "Scripts", "pip")
    else:
        activate_script = os.path.join(venv_name, "bin", "activate")
        pip_cmd = os.path.join(venv_name, "bin", "pip")
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        return False
    
    # Install ML-specific requirements if they exist
    ml_reqs = os.path.join("ml", "requirements.txt")
    if os.path.exists(ml_reqs):
        print("\nInstalling ML dependencies...")
        if not run_command(f"{pip_cmd} install -r {ml_reqs}"):
            return False
    
    return True

def setup_node_environment():
    """Set up Node.js environment for the frontend."""
    frontend_dir = "frontend"
    if not os.path.exists(frontend_dir):
        print("\nFrontend directory not found, skipping frontend setup.")
        return True
    
    print("\n=== Setting up Node.js environment ===")
    
    # Check if Node.js is installed
    if not run_command("node --version"):
        print("Node.js is not installed. Please install Node.js and try again.")
        return False
    
    # Install Node.js dependencies
    print("\nInstalling Node.js dependencies...")
    if not run_command("npm install", cwd=frontend_dir):
        return False
    
    return True

def setup_environment():
    """Main setup function."""
    print("=== Setting up Crop Health Prediction System ===")
    
    # Create necessary directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Set up Python environment
    if not setup_python_environment():
        print("\n❌ Failed to set up Python environment.")
        return False
    
    # Set up Node.js environment
    if not setup_node_environment():
        print("\n⚠️  Frontend setup had issues. You may need to set it up manually.")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and update with your configuration")
    print("2. Run the training script: python ml/train_plantvillage_fixed.py")
    print("3. Start the backend: python -m uvicorn backend.main:app --reload")
    print("4. In a new terminal, start the frontend: cd frontend && npm run dev")
    
    return True

if __name__ == "__main__":
    setup_environment()
