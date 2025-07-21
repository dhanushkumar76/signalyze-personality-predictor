"""
⚙️ Signalyze: Loss Function Registry for Multi-Trait Model

Supports:
- CategoricalCrossentropy
- Focal Loss (modular, optional per trait)

Used by train_model.py → model.compile(loss=get_loss_map(traits), ...)
"""

import tensorflow as tf

def categorical_loss():
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

def get_loss_map(trait_names):
    """
    Returns dict: {trait_output_name: loss_function}
    """
    loss_map = {}
    # Assign Categorical or Focal per trait (you can tune this manually)
    for trait in trait_names:
        if trait in ["Confidence", "Decision-Making"]:
            loss_map[f"{trait}_output"] = focal_loss()
        else:
            loss_map[f"{trait}_output"] = categorical_loss()
    return loss_map