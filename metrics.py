"""
PARTIE 2 - EXERCICE 2.2: Segmentation-Specific Metrics
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice Loss = 1 - Dice Coefficient
    Better for imbalanced segmentation tasks
    """
    return 1 - dice_coeff(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1.):
    """
    IoU (Intersection over Union) / Jaccard Index
    Formula: IoU = |A ∩ B| / |A ∪ B|
    """
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
    union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth), axis=-1)

if __name__ == '__main__':
    print("="*60)
    print("EXERCICE 2.2: Segmentation Metrics")
    print("="*60)
    
    # Test avec des données synthétiques
    print("\nTest des métriques avec données synthétiques...")
    
    # Cas 1: Prédiction parfaite
    y_true = tf.constant([[1., 1., 0., 0.], [1., 0., 0., 0.]])
    y_pred = tf.constant([[1., 1., 0., 0.], [1., 0., 0., 0.]])

    dice_val = dice_coeff(y_true, y_pred).numpy()
    iou_val = iou_metric(y_true, y_pred).numpy()
    
    print("\nCas 1 - Prédiction parfaite:")
    print(f"  Dice Coefficient: {dice_val:.4f}")
    print(f"  IoU Metric: {iou_val:.4f}")
    
    # Cas 2: Prédiction avec erreur
    y_pred2 = tf.constant([[1., 1., 0., 0.], [1., 1., 0., 0.]])  # 1 faux positif

    dice_val2 = dice_coeff(y_true, y_pred2).numpy()
    iou_val2 = iou_metric(y_true, y_pred2).numpy()
    
    print("\nCas 2 - Avec 1 faux positif:")
    print(f"  Dice Coefficient: {dice_val2:.4f}")
    print(f"  IoU Metric: {iou_val2:.4f}")
    
    # Cas 3: Prédiction aléatoire
    np.random.seed(42)
    y_pred3 = tf.constant(np.random.rand(2, 4), dtype=tf.float32)

    dice_val3 = dice_coeff(y_true, y_pred3).numpy()
    iou_val3 = iou_metric(y_true, y_pred3).numpy()
    
    print("\nCas 3 - Prédiction aléatoire:")
    print(f"  Dice Coefficient: {dice_val3:.4f}")
    print(f"  IoU Metric: {iou_val3:.4f}")
    
    print("\n✅ Exercice 2.2 terminé!")