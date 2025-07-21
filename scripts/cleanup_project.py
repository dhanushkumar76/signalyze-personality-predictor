"""
🧹 Cleanup Script for Signalyze Project
Removes all outdated logs, model files, and plots.
Safely preserves raw data files.

Run this to start your project with a clean slate.
"""

import os
import glob
import shutil

# --- Paths to Clean ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, '..')

def delete_files_and_folders():
    print("🧹 Starting project cleanup...")

    # Define directories to clean
    dirs_to_clean = [
        os.path.join(ROOT, 'model', 'evaluation_results'),
        os.path.join(ROOT, 'logs'),
    ]

    # Delete all files in specified directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}. Reason: {e}")

    # Delete specific top-level generated files
    files_to_delete = [
        os.path.join(ROOT, 'model', 'best_model.keras'),
        os.path.join(ROOT, 'model', 'training_log.csv'),
        os.path.join(ROOT, 'model', 'f1_accuracy_chart.png'),
        os.path.join(ROOT, 'model', 'metrics_report.csv')
    ]
    
    # Also delete any old confusion matrices
    conf_matrix_pattern = os.path.join(ROOT, 'model', 'conf_matrix_trait_*.png')
    for file in glob.glob(conf_matrix_pattern):
        files_to_delete.append(file)

    for file in files_to_delete:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Deleted: {file}")
        except FileNotFoundError:
            print(f"⚠️ Skipped (not found): {file}")
        except Exception as e:
            print(f"❌ Error deleting {file}: {e}")

    print("\n🧽 Cleanup complete. Your workspace is ready.")

if __name__ == "__main__":
    delete_files_and_folders()