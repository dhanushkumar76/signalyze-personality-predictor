import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import keras

@keras.saving.register_keras_serializable(package="custom_losses")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss for handling difficult traits"""
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_1(y_true, y_pred):
    """Weighted loss for trait 1"""
    return focal_loss(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_2(y_true, y_pred):
    """Weighted loss for trait 2"""
    return CategoricalCrossentropy()(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_3(y_true, y_pred):
    """Weighted loss for trait 3"""
    return focal_loss(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_4(y_true, y_pred):
    """Weighted loss for trait 4"""
    return CategoricalCrossentropy()(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_5(y_true, y_pred):
    """Weighted loss for trait 5"""
    return CategoricalCrossentropy()(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_6(y_true, y_pred):
    """Weighted loss for trait 6"""
    return focal_loss(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_7(y_true, y_pred):
    """Weighted loss for trait 7"""
    return focal_loss(y_true, y_pred)

@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_8(y_true, y_pred):
    """Weighted loss for trait 8"""
    return CategoricalCrossentropy()(y_true, y_pred)

# Dictionary of custom objects for model loading
CUSTOM_OBJECTS = {
    'focal_loss': focal_loss,
    'weighted_loss_trait_1': weighted_loss_trait_1,
    'weighted_loss_trait_2': weighted_loss_trait_2,
    'weighted_loss_trait_3': weighted_loss_trait_3,
    'weighted_loss_trait_4': weighted_loss_trait_4,
    'weighted_loss_trait_5': weighted_loss_trait_5,
    'weighted_loss_trait_6': weighted_loss_trait_6,
    'weighted_loss_trait_7': weighted_loss_trait_7,
    'weighted_loss_trait_8': weighted_loss_trait_8,
}