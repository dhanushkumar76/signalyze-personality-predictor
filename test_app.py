#!/usr/bin/env python3
"""
Signalyze App Test Script
========================
This script tests the main components of the Signalyze application.

Note to Maintainers:
====================
This script currently evaluates and reports only 4 traits for each signature:
- Ink Density
- Aspect Ratio
- Slant Angle
- Pen Pressure

Only 4 traits are used in all evaluation and reporting.
All trait-related code, comments, and docstrings have been updated to reflect this.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("   ✅ Streamlit")
    except ImportError as e:
        print(f"   ❌ Streamlit: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow ({tf.__version__})")
    except ImportError as e:
        print(f"   ❌ TensorFlow: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"   ✅ Pandas ({pd.__version__})")
    except ImportError as e:
        print(f"   ❌ Pandas: {e}")
        return False
    
    try:
        import cv2
        print(f"   ✅ OpenCV ({cv2.__version__})")
    except ImportError as e:
        print(f"   ❌ OpenCV: {e}")
        return False
    
    try:
        import sklearn
        print(f"   ✅ Scikit-learn ({sklearn.__version__})")
    except ImportError as e:
        print(f"   ❌ Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print(f"   ✅ Matplotlib ({matplotlib.__version__})")
    except ImportError as e:
        print(f"   ❌ Matplotlib: {e}")
        return False
    
    try:
        import plotly
        print(f"   ✅ Plotly ({plotly.__version__})")
    except ImportError as e:
        print(f"   ❌ Plotly: {e}")
        return False
    
    return True

def test_directories():
    """Test that all required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "data/preprocessed_images",
        "model", 
        "logs",
        "scripts",
        "pages",
        "utils"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory} (missing)")
            all_exist = False
    
    return all_exist

def test_core_files():
    """Test that core application files exist"""
    print("\n📄 Testing core files...")
    
    required_files = [
        "streamlit_app.py",
        "requirements.txt",
        "utils/logger.py",
        "pages/visual_traits_explainer.py",
        "pages/2_📁_Prediction_History.py",
        "scripts/3_train_model.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_image_processing():
    """Test image processing functions"""
    print("\n🖼️  Testing image processing...")
    
    try:
        # Create a dummy signature image
        dummy_img = np.ones((100, 200), dtype=np.uint8) * 255
        cv2.rectangle(dummy_img, (50, 30), (150, 70), 0, -1)  # Black rectangle
        
        # Test CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(dummy_img)
        print("   ✅ CLAHE processing")
        
        # Test resizing
        resized = cv2.resize(dummy_img, (128, 128))
        print("   ✅ Image resizing")
        
        # Test RGB conversion
        rgb_img = np.stack([resized] * 3, axis=-1)
        print("   ✅ RGB conversion")
        
        # Test basic trait extraction
        binary = (resized < 200).astype(np.uint8)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            ink_density = np.sum(binary) / (128 * 128)
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            height = y_coords.max() - y_coords.min()
            width = x_coords.max() - x_coords.min()
            aspect_ratio = height / width if width != 0 else 1.0
            print(f"   ✅ Trait extraction (density: {ink_density:.4f}, ratio: {aspect_ratio:.4f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return False

def test_logger():
    """Test logging functionality"""
    print("\n📝 Testing logger...")
    
    try:
        # Import logger
        sys.path.append('utils')
        from logger import log_prediction
        
        # Test logging
        test_traits = {"ink_density": 0.1234, "aspect_ratio": 0.8765, "slant_angle": 15.23}
        test_predictions = {"Confidence": "Agree (0.85)", "Emotional Stability": "Neutral (0.72)"}
        
        log_prediction("test_signature.png", test_traits, test_predictions)
        
        # Check if log file was created
        if os.path.exists("logs/prediction_log.csv"):
            print("   ✅ Logging functionality")
            return True
        else:
            print("   ❌ Log file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Logger test failed: {e}")
        return False

def test_streamlit_app():
    """Test that streamlit app can be parsed"""
    print("\n🌐 Testing Streamlit app structure...")
    
    try:
        # Try to parse the main app file
        with open("streamlit_app.py", 'r') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "st.set_page_config",
            "st.title", 
            "st.file_uploader",
            "extract_visual_traits",
            "load_model",
            "MODEL_PATH"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"   ❌ Missing components: {missing_components}")
            return False
        else:
            print("   ✅ Streamlit app structure")
            return True
            
    except Exception as e:
        print(f"   ❌ Streamlit app test failed: {e}")
        return False

def create_test_signature():
    """Create a test signature image"""
    print("\n🎨 Creating test signature...")
    
    try:
        # Create a more realistic signature-like image
        img = np.ones((100, 300), dtype=np.uint8) * 255
        
        # Draw signature-like curves
        pts1 = np.array([[50, 50], [80, 30], [120, 70], [150, 40], [180, 60], [220, 30], [250, 50]], np.int32)
        pts2 = np.array([[60, 65], [90, 80], [130, 60], [170, 75], [210, 55], [240, 70]], np.int32)
        
        cv2.polylines(img, [pts1], False, 0, 3)
        cv2.polylines(img, [pts2], False, 0, 2)
        
        # Add some dots and flourishes
        cv2.circle(img, (100, 45), 2, 0, -1)
        cv2.circle(img, (200, 35), 2, 0, -1)
        
        # Save test image
        os.makedirs("data/test_images", exist_ok=True)
        test_path = "data/test_images/test_signature.png"
        cv2.imwrite(test_path, img)
        
        print(f"   ✅ Test signature created: {test_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Test signature creation failed: {e}")
        return False

def run_tests():
    """Run all tests"""
    print("🚀 Starting Signalyze Application Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories), 
        ("Core Files", test_core_files),
        ("Image Processing", test_image_processing),
        ("Logger", test_logger),
        ("Streamlit App", test_streamlit_app),
        ("Test Signature", create_test_signature)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Signalyze is ready to run.")
        print("\n🚀 To start the app, run:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("   You may need to run setup.py first or install missing dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)