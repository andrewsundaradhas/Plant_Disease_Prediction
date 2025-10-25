# colab_setup.py
import os
from pyngrok import ngrok
import socket
import getpass
import time

def setup_colab_connection():
    # Get ngrok authtoken
    ngrok_token = getpass.getpass("Enter your ngrok authtoken (get it from https://dashboard.ngrok.com/get-started/your-authtoken): ")
    
    try:
        # Set ngrok authtoken
        ngrok.set_auth_token(ngrok_token)
        
        # Create a TCP tunnel for SSH
        print("\nCreating secure tunnel...")
        port = 22  # Default SSH port
        tunnel = ngrok.connect(port, "tcp")
        public_url = tunnel.public_url.replace("tcp://", "")
        host, port = public_url.split(":")
        
        print("\n" + "="*60)
        print("✅ Colab Connection Setup Complete")
        print("="*60)
        print("\n📡 SSH Connection Details:")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print("\n🔑 Keep this terminal running while training")
        print("="*60 + "\n")
        
        # Get the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Project directory: {project_dir}")
        
        # Keep the tunnel open
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nClosing connection...")
                ngrok.kill()
                break
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your ngrok token and internet connection.")

if __name__ == "__main__":
    print("="*60)
    print("🌐 Colab Remote Training Setup")
    print("="*60)
    print("This script will help you connect to Google Colab's GPU")
    print("1. First, get your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
    print("2. Run this script and enter your authtoken when prompted")
    print("3. Use the connection details in the Colab notebook")
    print("="*60 + "\n")
    
    try:
        setup_colab_connection()
    except KeyboardInterrupt:
        print("\n👋 Connection closed by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        print("\n✨ Setup completed. You can now run the Colab notebook.")
