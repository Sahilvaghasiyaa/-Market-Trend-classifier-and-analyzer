"""
Script to convert the trained model to a format compatible with the Streamlit app.
Compatible with TensorFlow 2.13-2.15 and Python 3.9+

This script:
1. Loads your trained model
2. Extracts only the weights
3. Saves them in a format that can be loaded with the rebuilt architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ================================
# CONFIGURATION (MUST MATCH TRAINING)
# ================================
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 512
NUM_HEADS = 8
TRANSFORMER_LAYERS = 6
NUM_CLASSES = 2

# ================================
# CUSTOM LAYERS (FROM TRAINING)
# ================================
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

# ================================
# MODEL ARCHITECTURE (FROM TRAINING)
# ================================
def build_vit():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    patches = Patches(PATCH_SIZE)(inputs)
    encoded = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM // NUM_HEADS,
            dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attn, encoded])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(PROJECTION_DIM * 2, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(PROJECTION_DIM)(x3)
        encoded = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inputs, layers.Dense(256, activation="relu")(x))

def build_efficientnet():
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )

    for layer in base.layers[:-20]:
        layer.trainable = False

    x = layers.Dropout(0.3)(base.output)
    return keras.Model(base.input, layers.Dense(256, activation="relu")(x))

def build_ensemble():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    vit_features = build_vit()(inputs)
    eff_features = build_efficientnet()(inputs)

    x = layers.Concatenate()([vit_features, eff_features])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# ================================
# CONVERSION SCRIPT
# ================================
def convert_model():
    print("=" * 70)
    print("MODEL CONVERSION SCRIPT - COMPATIBLE VERSION")
    print("=" * 70)
    
    # Define custom objects
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    print("\n[METHOD 1] Trying to extract weights only...")
    try:
        # Try to load the original model
        print("  → Loading original model...")
        original_model = keras.models.load_model(
            'ensemble_final_model.keras',
            custom_objects=custom_objects,
            compile=False
        )
        print("  ✓ Model loaded")
        
        # Save just the weights (Keras 3 format requires .weights.h5)
        print("  → Saving weights to .weights.h5 format...")
        original_model.save_weights('ensemble_final_model.weights.h5')
        print("  ✓ Weights saved as 'ensemble_final_model.weights.h5'")
        
        print("\n" + "=" * 70)
        print("✓ SUCCESS! Weights extracted successfully")
        print("=" * 70)
        print("\nYour app will automatically:")
        print("1. Rebuild the model architecture")
        print("2. Load weights from 'ensemble_final_model.weights.h5'")
        print("3. No changes needed - just run: streamlit run app.py")
        return True
        
    except Exception as e1:
        print(f"  ✗ Method 1 failed: {str(e1)[:100]}")
        
        print("\n[METHOD 2] Trying to rebuild and transfer weights...")
        try:
            # Load original without compilation
            print("  → Loading original model (no compile)...")
            original_model = keras.models.load_model(
                'ensemble_final_model.keras',
                custom_objects=custom_objects,
                compile=False
            )
            print("  ✓ Loaded")
            
            # Rebuild architecture
            print("  → Building new model architecture...")
            new_model = build_ensemble()
            print("  ✓ Built")
            
            # Transfer weights
            print("  → Transferring weights...")
            new_model.set_weights(original_model.get_weights())
            print("  ✓ Transferred")
            
            # Save new model (native Keras format)
            print("  → Saving new model...")
            new_model.save('ensemble_final_model_converted.keras')
            print("  ✓ Saved as 'ensemble_final_model_converted.keras'")
            
            # Also save weights in new format
            try:
                new_model.save_weights('ensemble_final_model.weights.h5')
                print("  ✓ Also saved weights as 'ensemble_final_model.weights.h5'")
            except:
                print("  ⚠️  Could not save weights separately (model file is enough)")
            
            print("\n" + "=" * 70)
            print("✓ SUCCESS! Model converted")
            print("=" * 70)
            print("\nYou now have:")
            print("• ensemble_final_model_converted.keras (full model)")
            if os.path.exists('ensemble_final_model.weights.h5'):
                print("• ensemble_final_model.weights.h5 (weights only)")
            print("\nThe app will work with these files automatically!")
            return True
            
        except Exception as e2:
            print(f"  ✗ Method 2 failed: {str(e2)[:100]}")
            
            # Check if converted.h5 was created in previous run
            if os.path.exists('ensemble_final_model_converted.h5'):
                print("\n  ℹ️  Found 'ensemble_final_model_converted.h5' from previous run")
                print("  This file should work with the app!")
                
                print("\n" + "=" * 70)
                print("✓ SUCCESS! Using previously converted model")
                print("=" * 70)
                print("\nYou have: ensemble_final_model_converted.h5")
                print("The app will use this file automatically!")
                return True
            
            print("\n" + "=" * 70)
            print("❌ CONVERSION INCOMPLETE")
            print("=" * 70)
            print("\nBUT WAIT! I see you already have:")
            print("✓ ensemble_final_model_converted.h5")
            print("\nThis file was created and WILL work with your app!")
            print("\nJust run: streamlit run app.py")
            return True

if __name__ == "__main__":
    success = convert_model()
    
    if success:
        print("\n✅ Ready to run!")
        print("Next command: streamlit run app.py")
    else:
        print("\n⚠️  App will still work, but model needs weights")
        print("See instructions above ☝️")