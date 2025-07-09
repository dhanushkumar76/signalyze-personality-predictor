#!/usr/bin/env python3
"""
Signalyze Setup Script
======================
This script helps set up the Signalyze personality predictor environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🖋️  SIGNALYZE SETUP                       ║
    ║              Signature-Based Personality Predictor           ║
    ║                                                              ║
    ║  This script will help you set up the complete environment  ║
    ║  for running Signalyze on your system.                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required. Current version:", f"{version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected - Compatible!")
        return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/preprocessed_images",
        "data/all_images", 
        "model",
        "logs",
        "model/evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Please check your internet connection and try again.")
        return False

def check_gpu_support():
    """Check for GPU support"""
    print("\n🎮 Checking GPU support...")
    
    try:
        import tensorflow as tf
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower).")
            print("   For faster training, consider using Google Colab with GPU.")
        
        # Check TensorFlow version
        print(f"   TensorFlow version: {tf.__version__}")
        
    except ImportError:
        print("⚠️  Could not import TensorFlow. Dependencies may not be installed correctly.")

def create_sample_data():
    """Create sample data files if they don't exist"""
    print("\n📊 Setting up sample data structure...")
    
    sample_responses_path = "data/form_responses.csv"
    sample_traits_path = "data/signature_traits.csv"
    
    if not os.path.exists(sample_responses_path):
        # Create minimal sample CSV structure
        sample_csv_content = """id,timestamp,Confidence,Emotional Stability,Sociability,Responsiveness,Concentration,Introversion,Creativity,Decision-Making
1,2024-01-01 10:00:00,Agree,Neutral,Agree,Disagree,Neutral,Agree,Strongly Agree,Neutral
2,2024-01-01 11:00:00,Neutral,Agree,Disagree,Agree,Strongly Agree,Neutral,Agree,Disagree"""
        
        with open(sample_responses_path, 'w') as f:
            f.write(sample_csv_content)
        print(f"   ✅ Created sample: {sample_responses_path}")
    
    if not os.path.exists(sample_traits_path):
        # Create minimal sample traits CSV
        sample_traits_content = """id,ink_density,aspect_ratio,slant_angle
1,0.1234,0.8765,15.23
2,0.2345,1.1234,-8.45"""
        
        with open(sample_traits_path, 'w') as f:
            f.write(sample_traits_content)
        print(f"   ✅ Created sample: {sample_traits_path}")

def print_next_steps():
    """Print instructions for next steps"""
    next_steps = """
    
    🎉 SETUP COMPLETE! 
    
    📋 Next Steps:
    
    1. 🚀 Run the Streamlit app:
       streamlit run streamlit_app.py
    
    2. 🌐 Open your browser to: http://localhost:8501
    
    3. 📸 Upload a signature image to test the system
    
    4. 🧠 To train your own model:
       python scripts/3_train_model.py
    
    5. 📊 Evaluate model performance:
       python scripts/evaluate_model.py
    
    ⚠️  Note: For training, you'll need:
       - Real signature images in data/preprocessed_images/
       - Corresponding survey responses in data/form_responses.csv
       - Visual traits in data/signature_traits.csv
    
    📖 For detailed instructions, see README.md
    
    🆘 Need help? Check the troubleshooting section in README.md
    """
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        print("   Please resolve the issues and run setup again.")
        sys.exit(1)
    
    # Check GPU support
    check_gpu_support()
    
    # Create sample data
    create_sample_data()
    
    # Print next steps
    print_next_steps()
    
    print("✅ Signalyze setup completed successfully!")

if __name__ == "__main__":
    main()