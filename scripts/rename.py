import os
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed_images")

# --- SCRIPT ---
print("Starting image renaming process...")

# Get a sorted list of all files in the directory
files = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Check if there are any images to rename
if not files:
    print("No images found in the preprocessed_images folder. Exiting.")
else:
    print(f"Found {len(files)} images to rename.")
    
    for i, old_name in enumerate(files, 1):
        # Create the new filename
        new_name = f"sample_{i}.jpg"
        
        old_path = os.path.join(PREPROCESSED_DIR, old_name)
        new_path = os.path.join(PREPROCESSED_DIR, new_name)
        
        try:
            shutil.move(old_path, new_path)
        except Exception as e:
            print(f"Error renaming {old_name} to {new_name}: {e}")
            
    print("\n✅ Renaming complete!")
    print(f"All {len(files)} images have been renamed to the 'sample_X.jpg' format.")