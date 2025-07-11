# === scripts/evaluate_model.py ===
"""
Comprehensive model evaluation script for Signalyze personality predictor (4 traits version).
Generates detailed performance metrics, F1-scores, and accuracy analysis for:
- Confidence
- Emotional Stability
- Creativity
- Decision-Making
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import CategoricalCrossentropy

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
CSV_PATH = os.path.join(BASE_DIR, "data", "form_responses.csv")
FEATURE_CSV = os.path.join(BASE_DIR, "data", "signature_traits.csv")
IMG_FOLDER = os.path.join(BASE_DIR, "data", "preprocessed_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "evaluation_results")

IMAGE_SIZE = (128, 128)
NUM_TRAITS = 4  # Only 4 traits used in the current model
NUM_CLASSES = 3
LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Creativity", "Decision-Making",
]
CLASS_NAMES = ["Disagree", "Neutral", "Agree"]

# --- Custom loss functions for model loading ---
@register_keras_serializable()
def focal_loss_inner(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

def make_registered_loss(trait_name, base_loss):
    @register_keras_serializable(name=f"weighted_loss_{trait_name}")
    def trait_loss(y_true, y_pred):
        return base_loss(y_true, y_pred)
    trait_loss.__name__ = f"weighted_loss_{trait_name}"
    return trait_loss

loss_map = {}
for i in range(NUM_TRAITS):
    trait_key = f"trait_{i+1}"
    use_focal = i == 2  # Only 'Creativity' uses focal loss
    base = focal_loss_inner if use_focal else CategoricalCrossentropy()
    loss_map[trait_key] = make_registered_loss(trait_key, base)

custom_objects = {f"weighted_loss_trait_{i+1}": loss_map[f"trait_{i+1}"] for i in range(NUM_TRAITS)}

def load_test_data():
    """Load and preprocess test data"""
    print("📂 Loading evaluation data...")
    
    # Check if files exist
    for path, name in [(CSV_PATH, "Survey responses"), (FEATURE_CSV, "Visual traits"), (IMG_FOLDER, "Images")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")
    
    # Load data
    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    print(f"   Found {len(files)} images")
    print(f"   Survey responses: {len(df)} rows")
    print(f"   Visual traits: {len(traits_df)} rows")
    
    # Align data sizes
    min_size = min(len(files), len(df), len(traits_df))
    df, traits_df, files = df.iloc[:min_size], traits_df.iloc[:min_size], files[:min_size]
    
    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]
    
    print("⚙️  Processing images...")
    for i, fname in enumerate(files):
        if i % 50 == 0:
            print(f"   Processing {i+1}/{len(files)}")
            
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack([img]*3, axis=-1)
        img = preprocess_input(img.astype(np.float32))
        X.append(img)
        
        Y_feats.append([
            traits_df.iloc[i]["ink_density"],
            traits_df.iloc[i]["aspect_ratio"],
            traits_df.iloc[i]["slant_angle"]
        ])
        
        for j in range(NUM_TRAITS):
            val = str(df.iloc[i, -NUM_TRAITS + j]).strip()
            y_outputs[j].append(LIKERT_MAP.get(val, 2))
    
    print(f"✅ Loaded {len(X)} samples")
    
    # Split data (same random state as training)
    X_train, X_test, Y_train, Y_test = train_test_split(
        np.array(X), np.array(Y_feats), test_size=0.15, random_state=42
    )
    
    y_test = []
    for y in y_outputs:
        _, y_t = train_test_split(y, test_size=0.15, random_state=42)
        y_test.append(np.array(y_t))
    
    return X_test, Y_test, y_test

def evaluate_model(model, X_test, Y_test, y_test):
    """Comprehensive model evaluation"""
    print("\n🧠 Making predictions...")
    
    # Get predictions
    predictions = model.predict([X_test, Y_test], verbose=1)
    
    # Convert to class predictions
    y_pred = [np.argmax(pred, axis=1) for pred in predictions]
    y_true = [np.array(y) for y in y_test]
    
    print("\n📊 Calculating metrics...")
    
    results = {
        'trait_names': TRAIT_NAMES,
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': [],
        'precision_macro': [],
        'recall_macro': [],
        'confusion_matrices': []
    }
    
    # Calculate metrics for each trait
    for i in range(NUM_TRAITS):
        # Basic metrics
        accuracy = accuracy_score(y_true[i], y_pred[i])
        f1_macro = f1_score(y_true[i], y_pred[i], average='macro')
        f1_weighted = f1_score(y_true[i], y_pred[i], average='weighted')
        precision_macro = precision_score(y_true[i], y_pred[i], average='macro')
        recall_macro = recall_score(y_true[i], y_pred[i], average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(y_true[i], y_pred[i])
        
        # Store results
        results['accuracy'].append(accuracy)
        results['f1_macro'].append(f1_macro)
        results['f1_weighted'].append(f1_weighted)
        results['precision_macro'].append(precision_macro)
        results['recall_macro'].append(recall_macro)
        results['confusion_matrices'].append(cm)
        
        print(f"   {TRAIT_NAMES[i]:20} | Acc: {accuracy:.3f} | F1: {f1_macro:.3f} | Prec: {precision_macro:.3f} | Rec: {recall_macro:.3f}")
    
    return results

def generate_detailed_report(results):
    """Generate detailed evaluation report"""
    print("\n📋 Generating detailed report...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Trait': results['trait_names'],
        'Accuracy': results['accuracy'],
        'F1_Macro': results['f1_macro'],
        'F1_Weighted': results['f1_weighted'],
        'Precision_Macro': results['precision_macro'],
        'Recall_Macro': results['recall_macro']
    })
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary saved to: {summary_path}")
    
    # Calculate overall metrics
    overall_metrics = {
        'Mean_Accuracy': np.mean(results['accuracy']),
        'Mean_F1_Macro': np.mean(results['f1_macro']),
        'Mean_F1_Weighted': np.mean(results['f1_weighted']),
        'Mean_Precision': np.mean(results['precision_macro']),
        'Mean_Recall': np.mean(results['recall_macro']),
        'Std_Accuracy': np.std(results['accuracy']),
        'Std_F1_Macro': np.std(results['f1_macro'])
    }
    
    # Save overall metrics
    overall_df = pd.DataFrame([overall_metrics])
    overall_path = os.path.join(OUTPUT_DIR, "overall_metrics.csv")
    overall_df.to_csv(overall_path, index=False)
    
    return summary_df, overall_metrics

def create_visualizations(results, summary_df):
    """Create comprehensive visualizations"""
    print("\n📈 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Performance metrics comparison
    plt.subplot(3, 3, 1)
    x = np.arange(len(TRAIT_NAMES))
    width = 0.2
    
    plt.bar(x - width, results['accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x, results['f1_macro'], width, label='F1-Macro', alpha=0.8)
    plt.bar(x + width, results['precision_macro'], width, label='Precision', alpha=0.8)
    
    plt.xlabel('Personality Traits')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Trait')
    plt.xticks(x, [name[:8] for name in TRAIT_NAMES], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1-Score distribution
    plt.subplot(3, 3, 2)
    plt.hist(results['f1_macro'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(results['f1_macro']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(results["f1_macro"]):.3f}')
    plt.xlabel('F1-Score (Macro)')
    plt.ylabel('Frequency')
    plt.title('F1-Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Accuracy vs F1-Score scatter
    plt.subplot(3, 3, 3)
    plt.scatter(results['accuracy'], results['f1_macro'], alpha=0.7, s=100)
    for i, trait in enumerate(TRAIT_NAMES):
        plt.annotate(trait[:8], (results['accuracy'][i], results['f1_macro'][i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Accuracy')
    plt.ylabel('F1-Score (Macro)')
    plt.title('Accuracy vs F1-Score')
    plt.grid(True, alpha=0.3)
    
    # 4-9. Confusion matrices (2x3 grid)
    for i in range(NUM_TRAITS):
        plt.subplot(3, 3, 4 + (i % 6))
        if i < 6:  # First 6 traits
            cm = results['confusion_matrices'][i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.title(f'{TRAIT_NAMES[i][:12]}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
    
    plt.tight_layout()
    
    # Save main visualization
    viz_path = os.path.join(OUTPUT_DIR, "evaluation_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate confusion matrix grid for remaining traits
    if NUM_TRAITS > 6:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i in range(6, min(8, NUM_TRAITS)):
            cm = results['confusion_matrices'][i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[i-6])
            axes[i-6].set_title(f'{TRAIT_NAMES[i]}')
            axes[i-6].set_xlabel('Predicted')
            axes[i-6].set_ylabel('True')
        
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices_extended.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   Visualizations saved to: {OUTPUT_DIR}")

def print_summary_report(summary_df, overall_metrics):
    """Print comprehensive summary report"""
    print("\n" + "="*80)
    print("🎯 SIGNALYZE MODEL EVALUATION REPORT")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Mean Accuracy:     {overall_metrics['Mean_Accuracy']:.3f} ± {overall_metrics['Std_Accuracy']:.3f}")
    print(f"   Mean F1-Score:     {overall_metrics['Mean_F1_Macro']:.3f} ± {overall_metrics['Std_F1_Macro']:.3f}")
    print(f"   Mean Precision:    {overall_metrics['Mean_Precision']:.3f}")
    print(f"   Mean Recall:       {overall_metrics['Mean_Recall']:.3f}")
    
    print(f"\n🏆 TOP PERFORMING TRAITS:")
    top_traits = summary_df.nlargest(3, 'F1_Macro')[['Trait', 'Accuracy', 'F1_Macro']]
    for _, row in top_traits.iterrows():
        print(f"   {row['Trait']:20} | Acc: {row['Accuracy']:.3f} | F1: {row['F1_Macro']:.3f}")
    
    print(f"\n⚠️  CHALLENGING TRAITS:")
    bottom_traits = summary_df.nsmallest(3, 'F1_Macro')[['Trait', 'Accuracy', 'F1_Macro']]
    for _, row in bottom_traits.iterrows():
        print(f"   {row['Trait']:20} | Acc: {row['Accuracy']:.3f} | F1: {row['F1_Macro']:.3f}")
    
    print(f"\n📈 RECOMMENDATIONS:")
    if overall_metrics['Mean_Accuracy'] < 0.7:
        print("   • Consider data augmentation or model architecture improvements")
    if overall_metrics['Std_F1_Macro'] > 0.1:
        print("   • High variance between traits - consider trait-specific tuning")
    if overall_metrics['Mean_F1_Macro'] < 0.6:
        print("   • Low F1-scores indicate class imbalance - consider focal loss or resampling")
    
    print("="*80)

def main():
    """Main evaluation pipeline"""
    print("🚀 Starting Signalyze Model Evaluation")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please train the model first using scripts/3_train_model.py")
        return
    
    try:
        # Load model
        print(f"🧠 Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✅ Model loaded successfully")
        
        # Load test data
        X_test, Y_test, y_test = load_test_data()
        
        # Evaluate model
        results = evaluate_model(model, X_test, Y_test, y_test)
        
        # Generate reports
        summary_df, overall_metrics = generate_detailed_report(results)
        
        # Create visualizations
        create_visualizations(results, summary_df)
        
        # Print summary
        print_summary_report(summary_df, overall_metrics)
        
        print(f"\n✅ Evaluation completed! Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()