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
IMG_SIZE = 224

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

CLASSES = ["buyinput", "sellinput"]

PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 512
NUM_HEADS = 8
TRANSFORMER_LAYERS = 6

# ================================
# CUSTOM LAYERS
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
# PREPROCESSING
# ================================
def preprocess_image_for_model(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def encode_image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ================================
# MODEL LOADING (FIXED)
# ================================
@st.cache_resource
def load_trained_model():
    custom_objects = {
        "Patches": Patches,
        "PatchEncoder": PatchEncoder
    }

    # Method 1: full .keras
    if os.path.exists("ensemble_final_model.keras"):
        try:
            st.info("🔄 Loading full model (.keras)...")
            model = keras.models.load_model(
                "ensemble_final_model.keras",
                custom_objects=custom_objects,
                compile=False
            )
            st.success("✅ Model loaded (.keras)")
            return model
        except Exception as e:
            st.warning(f"⚠️ .keras load failed: {str(e)[:120]}")

    # Method 2: rebuild + weights
    for wf in [
        "ensemble_final_model.weights.h5",
        "ensemble_final_model_converted.h5"
    ]:
        if os.path.exists(wf):
            try:
                st.info(f"🔄 Loading weights ({wf})...")
                model = build_ensemble()
                model.load_weights(wf)
                st.success(f"✅ Model loaded from weights: {wf}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed {wf}: {str(e)[:120]}")

    st.error("❌ Model could not be loaded.")
    return None

# ================================
# GPT-4o VISION ANALYSIS
# ================================
def analyze_with_gpt4_vision(image):
    base64_image = encode_image_to_base64(image)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert technical analyst."},
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

# ================================
# HYBRID ANALYSIS
# ================================
def perform_hybrid_analysis(image, model):
    gpt_result = analyze_with_gpt4_vision(image)
    pre = preprocess_image_for_model(image)
    pred = model.predict(pre, verbose=0)
    bias = CLASSES[np.argmax(pred[0])]
    gpt_result["model_bias"] = "Bullish" if bias == "buyinput" else "Bearish"
    return gpt_result

# ================================
# STREAMLIT UI
# ================================
def main():
    st.set_page_config("AI Chart Analysis", "📊", layout="wide")

    model = load_trained_model()
    if model is None:
        st.stop()

    uploaded = st.file_uploader(
        "Upload candlestick chart",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Chart", use_container_width=True)

        if st.button("🔬 Analyze Chart", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = perform_hybrid_analysis(image, model)
            st.json(result)


if __name__ == "__main__":
    main()
