"""
PARTIE 2 - EXERCICE 2.1: Implementing the U-Net Architecture
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

def conv_block(input_tensor, num_filters):
    """
    Core convolutional block: Conv2D -> BatchNorm -> ReLU (x2)
    """
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


def build_unet(input_shape=(128, 128, 1)):
    """
    U-Net Architecture Implementation
    """
    inputs = keras.Input(input_shape)
    
    # ========== ENCODER PATH (Contracting) ==========
    # Level 1
    c1 = conv_block(inputs, 32)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    # Level 2
    c2 = conv_block(p1, 64)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Level 3
    c3 = conv_block(p2, 128)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    # ========== BRIDGE/BOTTLENECK ==========
    b = conv_block(p3, 256)
    
    # ========== DECODER PATH (Expansive) ==========
    # Level 1: Upsampling + Skip Connection avec c3
    u1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = keras.layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 128)
    
    # Level 2: Upsampling + Skip Connection avec c2
    u2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)
    
    # Level 3: Upsampling + Skip Connection avec c1
    u3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = keras.layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 32)
    
    # ========== OUTPUT LAYER ==========
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)
    
    return keras.Model(inputs=[inputs], outputs=[outputs], name='U-Net')


if __name__ == '__main__':
    print("="*60)
    print("EXERCICE 2.1: U-Net Architecture")
    print("="*60)
    
    # Construction du modèle
    model = build_unet(input_shape=(128, 128, 1))
    
    # Affichage du résumé
    print("\nArchitecture U-Net:")
    model.summary()
    
    print(f"\nNombre total de paramètres: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    print("\n✅ Exercice 2.1 terminé!")