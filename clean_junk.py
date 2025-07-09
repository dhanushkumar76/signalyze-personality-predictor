import os
import glob
import shutil

# Root project folder
ROOT = "D:/signalyze/signalyze-personality-predictor"

# 1. Remove __pycache__ folders
def delete_pycache_dirs():
    for root, dirs, files in os.walk(ROOT):
        for d in dirs:
            if d == "__pycache__":
                full_path = os.path.join(root, d)
                shutil.rmtree(full_path)
                print(f"🗑️ Deleted __pycache__: {full_path}")

# 2. Remove loss curve and confusion matrices
def delete_generated_visuals():
    model_dir = os.path.join(ROOT, "model")
    if os.path.exists(model_dir):
        for f in glob.glob(os.path.join(model_dir, "loss_curve.png")):
            os.remove(f)
            print(f"🗑️ Deleted loss curve: {f}")
        for f in glob.glob(os.path.join(model_dir, "conf_matrix_trait_*.png")):
            os.remove(f)
            print(f"🗑️ Deleted confusion matrix: {f}")

# 3. Remove unnecessary OS/editor junk files
def delete_temp_files():
    patterns = ["*.tmp", "*.log~", "*.bak", "*.copy", ".DS_Store", "Thumbs.db"]
    for root, dirs, files in os.walk(ROOT):
        for pattern in patterns:
            for f in glob.glob(os.path.join(root, pattern)):
                os.remove(f)
                print(f"🗑️ Deleted temp/junk file: {f}")

# Run cleanups (excludes training_log.csv and prediction_log.csv)
delete_pycache_dirs()
delete_generated_visuals()
delete_temp_files()

print("\n✅ Cleanup complete. Logs and model preserved.")
