"""
🧹 Cleanup Script for Signalyze Project
Removes outdated logs, model files, plots, and CSVs related to previous 8-trait pipeline.
Safely preserves images and raw form responses.

Run before retraining and regenerating visuals.
"""

import os

# === Paths to Clean ===
FILES_TO_DELETE = [
    "logs/prediction_log.csv",
    "model/training_log.csv",
    "model/best_model.keras",
    "model/loss_curve.png",
    "model/f1_accuracy_chart.png",
    "model/final_accuracy_bar.png",
    "model/metrics_report.csv",
    "model/evaluation_visualization.png",  # if exists
    "model/evaluation_results/evaluation_summary.csv",
    "model/evaluation_results/overall_metrics.csv",
]

# Delete confusion matrices
for i in range(1, 9):
    conf_path = f"model/conf_matrix_trait_{i}.png"
    if os.path.exists(conf_path):
        FILES_TO_DELETE.append(conf_path)

# === Execute Deletion ===
for file in FILES_TO_DELETE:
    try:
        os.remove(file)
        print(f"✅ Deleted: {file}")
    except FileNotFoundError:
        print(f"⚠️ Skipped (not found): {file}")
    except Exception as e:
        print(f"❌ Error deleting {file}: {e}")

print("\n🧽 Cleanup complete. Your workspace is ready for 4-trait regeneration.")