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
    ║                    🖋️  SIGNALYZE SETUP                       ║
    ║              Signature-Based Personality Predictor           ║
    ║                                                              ║
    ║    This script will help you set up the complete environment  ║
    ║    for running Signalyze on your system.                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ is required. Current version:", f"{version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again.")
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
        print(f"   ✅ Created: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Please check your internet connection and try again.")
        return False

def check_gpu_support():
    """Check for GPU support"""
    print("\n🎮 Checking GPU support...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower).")
            print("   For faster training, consider using Google Colab with GPU.")
        
        print(f"   TensorFlow version: {tf.__version__}")
        
    except ImportError:
        print("⚠️  Could not import TensorFlow. Dependencies may not be installed correctly.")

def create_sample_data():
    """Create sample data files if they don't exist"""
    print("\n📊 Setting up sample data structure...")
    
    # FIX: Create a single, consistent master CSV
    master_csv_path = "data/handwriting_personality_large_dataset.csv"
    
    if not os.path.exists(master_csv_path):
        sample_csv_content = """Handwriting_Sample,Writing_Speed_wpm,Openness,Conscientiousness,Extraversion,Agreeableness,Neuroticism
sample_1.jpg,60,0.35,0.40,0.72,0.45,0.25
sample_2.jpg,32,0.73,0.05,0.35,0.52,0.66
sample_3.jpg,10,0.83,0.16,0.16,0.81,0.68"""
        
        with open(master_csv_path, 'w') as f:
            f.write(sample_csv_content)
        print(f"   ✅ Created consistent sample: {master_csv_path}")
    
    # FIX: Remove creation of old CSV files
    old_csv_path = "data/form_responses.csv"
    if os.path.exists(old_csv_path):
        os.remove(old_csv_path)
        print(f"   ✅ Removed old file: {old_csv_path}")

def print_next_steps():
    """Print instructions for next steps"""
    next_steps = """
    
    🎉 SETUP COMPLETE! 
    
    📋 Next Steps:
    
    1. 📥 Get your dataset: Download the Kaggle dataset and place the images in `data/all_images/`
    
    2. 📝 Run the preprocessing script:
        python scripts/2_preprocess_all_images.py
    
    3. 🖼️ Run the renaming script to align images with the CSV:
        python scripts/rename_preprocessed_images.py
    
    4. 🧠 Start training your model with the new data:
        python scripts/3_train_model.py
    
    5. 🚀 Run the Streamlit app:
        streamlit run streamlit_app.py
    
    📖 For detailed instructions, see README.md
    
    🆘 Need help? Check the troubleshooting section in README.md
    """
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation.")
        print("   Please resolve the issues and run setup again.")
        sys.exit(1)
    
    check_gpu_support()
    
    create_sample_data()
    
    print_next_steps()
    
    print("✅ Signalyze setup completed successfully!")

if __name__ == "__main__":
    main()