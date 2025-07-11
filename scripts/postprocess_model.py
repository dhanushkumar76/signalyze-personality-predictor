# postprocess_model.py
"""
Standalone script to generate analytics/plots from training logs and metrics for Signalyze (no training).

Only 4 traits are used in all post-processing analytics and plots.
All trait-related code, comments, and docstrings have been updated to reflect this.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")
TRAIT_NAMES = ["Confidence", "Emotional Stability", "Creativity", "Decision-Making"]
NUM_TRAITS = 4

if not os.path.exists(LOG_PATH):
    print(f"No training log found at {LOG_PATH}. Run training first.")
    exit(1)

df_log = pd.read_csv(LOG_PATH)

# --- Bar chart: Final accuracy by trait ---
acc_cols = [col for col in df_log.columns if col.endswith('_accuracy') and not col.startswith('val_')][:NUM_TRAITS]
val_acc_cols = [col for col in df_log.columns if col.startswith('val_') and col.endswith('_accuracy')][:NUM_TRAITS]
if acc_cols and val_acc_cols and len(acc_cols) == NUM_TRAITS and len(val_acc_cols) == NUM_TRAITS:
    train_acc = df_log[acc_cols].iloc[-1].values
    val_acc = df_log[val_acc_cols].iloc[-1].values
    acc_df = pd.DataFrame({
        'Trait': TRAIT_NAMES,
        'Train': train_acc,
        'Val': val_acc
    }).set_index('Trait')
    plt.figure(figsize=(8, 5))
    acc_df.plot(kind='bar')
    plt.title('Final Accuracy by Trait (Combined)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'final_accuracy_bar.png'), dpi=300)
    plt.close()
    print('Saved: final_accuracy_bar.png')

# --- Scatterplot: Accuracy vs F1-Score (if available) ---
f1_cols = [col for col in df_log.columns if col.endswith('_f1') and not col.startswith('val_')][:NUM_TRAITS]
val_f1_cols = [col for col in df_log.columns if col.startswith('val_') and col.endswith('_f1')][:NUM_TRAITS]
if f1_cols and val_f1_cols and len(f1_cols) == NUM_TRAITS and len(val_f1_cols) == NUM_TRAITS:
    train_f1 = df_log[f1_cols].iloc[-1].values
    val_f1 = df_log[val_f1_cols].iloc[-1].values
    scatter_df = pd.DataFrame({
        'Trait': TRAIT_NAMES,
        'Train Accuracy': train_acc,
        'Val Accuracy': val_acc,
        'Train F1': train_f1,
        'Val F1': val_f1
    })
    plt.figure(figsize=(7, 5))
    plt.scatter(scatter_df['Val Accuracy'], scatter_df['Val F1'])
    for i, txt in enumerate(scatter_df['Trait']):
        plt.annotate(txt, (scatter_df['Val Accuracy'][i], scatter_df['Val F1'][i]))
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Validation F1-Score')
    plt.title('Validation Accuracy vs F1-Score by Trait (Scatterplot)')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'accuracy_vs_f1_scatter.png'), dpi=300)
    plt.close()
    print('Saved: accuracy_vs_f1_scatter.png')

print('✅ Post-processing analytics and plots generated.')
