"""
PARTIE 3 - EXERCICE 3.2: Conv3D Block and Engineering Discipline
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import mlflow
import numpy as np


def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    """
    Simple Conv3D block for volumetric data (CT scans, MRI)
    Input: D × H × W × C (Depth, Height, Width, Channels)
    """
    inputs = keras.Input(input_shape)
    
    # Conv3D Block 1: 16 filters, 3×3×3 kernel
    x = keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    # Conv3D Block 2: 32 filters, 3×3×3 kernel
    x = keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool3D((2, 2, 2))(x)
    
    # Global pooling and output
    x = keras.layers.GlobalAveragePooling3D()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='Conv3D_Model')


if __name__ == '__main__':
    print("="*60)
    print("EXERCICE 3.2: Conv3D for Volumetric Data")
    print("="*60)
    
    # Setup MLflow experiment
    mlflow.set_experiment("3D_Volumetric_Analysis")
    
    with mlflow.start_run(run_name="Conv3D_Baseline"):
        print("\nBuilding Conv3D model...")
        model_3d = simple_conv3d_block(input_shape=(32, 32, 32, 1))
        
        # Log model architecture
        model_config = model_3d.to_json()
        mlflow.log_text(model_config, "model_architecture.json")
        
        # Log hyperparameters
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        mlflow.log_param("input_shape", "32×32×32×1")
        mlflow.log_param("kernel_size", "3×3×3")
        
        # Display architecture
        print("\nConv3D Architecture:")
        print("-"*60)
        model_3d.summary()
        
        print(f"\nTotal parameters: {model_3d.count_params():,}")
        
        # Simulate training metrics
        print("\n\nSimulating training (5 epochs)...")
        print("-"*60)
        
        simulated_metrics = [
            (0.693, 0.701),  # epoch 0
            (0.520, 0.542),  # epoch 1
            (0.387, 0.419),  # epoch 2
            (0.298, 0.358),  # epoch 3
            (0.245, 0.312),  # epoch 4
        ]
        
        for epoch, (train_loss, val_loss) in enumerate(simulated_metrics):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        final_val_loss = simulated_metrics[-1][1]
        mlflow.log_metric("final_val_loss", final_val_loss)
        
        print(f"\n{'='*60}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print('='*60)
        
    print("\n✅ Exercice 3.2 terminé!")
    print("\nView results in MLflow UI:")
    print("  → Run: mlflow ui")
    print("  → Open: http://localhost:5000")