import os
import io
import base64
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from openai import OpenAI

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "ensemble_final_model.keras"
IMG_SIZE = 224  # CRITICAL: Must match training size

# ✅ Streamlit Cloud–safe API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Model classes (must match training order)
CLASSES = ["buyinput", "sellinput"]

# Vision Transformer parameters (MUST MATCH TRAINING)
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 512
NUM_HEADS = 8
TRANSFORMER_LAYERS = 6

# ================================
# CUSTOM LAYERS (FROM TRAINING CODE)
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
# MODEL BUILDING
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

    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# ================================
# IMAGE PREPROCESSING
# ================================
def preprocess_image_for_model(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_trained_model():
    custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}

    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
            return model
        except Exception:
            pass

    st.error("❌ Model could not be loaded.")
    return None

# ================================
# GPT-4 VISION ANALYSIS
# ================================
def analyze_with_gpt4_vision(image, is_trend_check=True):
    base64_image = encode_image_to_base64(image)

    system_prompt = """You are an expert technical analyst with deep knowledge of price action, candlestick patterns, support/resistance, Fibonacci levels, and market structure. Analyze the candlestick chart image comprehensively and respond strictly in JSON as instructed."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this candlestick chart comprehensively."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )

        content = response.choices[0].message.content

        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

# ================================
# HYBRID ANALYSIS
# ================================
def perform_hybrid_analysis(image, model):
    gpt_analysis = analyze_with_gpt4_vision(image)
    if "error" in gpt_analysis:
        return gpt_analysis

    preprocessed = preprocess_image_for_model(image)
    prediction = model.predict(preprocessed, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])

    model_label = CLASSES[predicted_class_idx]
    model_bias = "Bullish" if model_label == "buyinput" else "Bearish"

    gpt_analysis["model_bias"] = model_bias
    return gpt_analysis

# ================================
# STREAMLIT UI
# ================================
def main():
    st.set_page_config(page_title="AI Chart Analysis", page_icon="📊", layout="wide")

    model = load_trained_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a candlestick chart image",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chart", use_container_width=True)

        if st.button("🔬 Analyze Chart", type="primary", use_container_width=True):
            results = perform_hybrid_analysis(image, model)
            st.json(results)


if __name__ == "__main__":
    main()
