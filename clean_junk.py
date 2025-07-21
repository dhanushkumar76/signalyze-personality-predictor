import os
import shutil
from datetime import datetime

# === Paths to clear ===
training_log = "model/training_log.csv"
metrics_report = "model/metrics_report.csv"
eval_results_dir = "model/evaluation_results"

def safe_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Removed: {file_path}")

def safe_clear_dir(folder_path):
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if fname.endswith(".png") or fname.endswith(".csv"):
                os.remove(fpath)
                print(f"✅ Cleared: {fpath}")

def main():
    print("🧹 Starting cleanup...")

    safe_remove(training_log)
    safe_remove(metrics_report)
    safe_clear_dir(eval_results_dir)

    print("✨ Logs cleared. Ready to retrain or evaluate!")

if __name__ == "__main__":
    main()