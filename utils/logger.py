# === utils/logger.py ===
import os
import csv
from datetime import datetime

def log_prediction(filename, traits, predictions, log_file='logs/prediction_log.csv'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "ink_density": traits.get("ink_density", 0),
        "aspect_ratio": traits.get("aspect_ratio", 0),
        "slant_angle": traits.get("slant_angle", 0)
    }

    for trait_name, result in predictions.items():
        row[trait_name] = result

    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
