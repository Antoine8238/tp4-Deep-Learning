"""
PARTIE 2: Training U-Net with MLflow Tracking
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

# Import from previous exercises
from unet import build_unet
from metrics import dice_coeff, dice_loss, iou_metric


def generate_synthetic_data(num_samples=200, img_size=128):
    """
    Generate synthetic medical segmentation data
    Simulates images with circular structures (organs/tumors)
    """
    print(f"Generating {num_samples} synthetic images ({img_size}×{img_size})...")
    
    X = np.zeros((num_samples, img_size, img_size, 1))
    Y = np.zeros((num_samples, img_size, img_size, 1))
    
    for i in range(num_samples):
        # Background with gaussian noise
        X[i, :, :, 0] = np.random.randn(img_size, img_size) * 0.1 + 0.3
        
        # Add 1-3 circular objects
        num_objects = np.random.randint(1, 4)
        for _ in range(num_objects):
            center_x = np.random.randint(20, img_size - 20)
            center_y = np.random.randint(20, img_size - 20)
            radius = np.random.randint(10, 25)
            
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            X[i, mask, 0] += np.random.rand() * 0.5 + 0.3
            Y[i, mask, 0] = 1
    
    X = np.clip(X, 0, 1)
    
    print(f"  Foreground ratio: {Y.mean()*100:.1f}%")
    
    return X.astype(np.float32), Y.astype(np.float32)


def train_unet(model, X_train, Y_train, X_val, Y_val, 
               run_name, loss_func, epochs=30):
    """
    Train U-Net with MLflow tracking
    """
    mlflow.set_experiment("TP4_Segmentation_Medical")
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print('='*60)
        
        # Log hyperparameters
        mlflow.log_param("architecture", "U-Net")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", loss_func.__name__)
        mlflow.log_param("learning_rate", 1e-4)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 16)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss_func,
            metrics=[dice_coeff, iou_metric]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.5,
                verbose=1
            )
        ]
        
        # Training
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log final metrics
        final_dice = history.history['val_dice_coeff'][-1]
        final_iou = history.history['val_iou_metric'][-1]
        final_loss = history.history['val_loss'][-1]
        
        mlflow.log_metric("final_val_dice", final_dice)
        mlflow.log_metric("final_val_iou", final_iou)
        mlflow.log_metric("final_val_loss", final_loss)
        
        # Log metrics per epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_dice", history.history['val_dice_coeff'][epoch], step=epoch)
            mlflow.log_metric("val_iou", history.history['val_iou_metric'][epoch], step=epoch)
        
        print(f"\n{'='*60}")
        print("Final Results:")
        print(f"  Dice Coefficient: {final_dice:.4f}")
        print(f"  IoU Metric: {final_iou:.4f}")
        print(f"  Loss: {final_loss:.4f}")
        print('='*60)
        
        return history


if __name__ == '__main__':
    print("="*60)
    print("PARTIE 2: U-Net Training with MLflow")
    print("="*60)
    
    # Generate data
    X, Y = generate_synthetic_data(num_samples=200, img_size=128)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {X_train.shape[0]} images")
    print(f"  Val: {X_val.shape[0]} images")
    
    # Build model
    print("\nBuilding U-Net model...")
    model = build_unet(input_shape=(128, 128, 1))
    
    # Train
    history = train_unet(
        model, X_train, Y_train, X_val, Y_val,
        run_name="UNet_DiceLoss_Adam",
        loss_func=dice_loss,
        epochs=30
    )
    
    print("\n✅ Training complete!")
    print("\nView results in MLflow UI:")
    print("  → Run: mlflow ui")
    print("  → Open: http://localhost:5000")