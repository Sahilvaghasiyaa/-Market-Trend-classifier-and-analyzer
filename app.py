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
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="AI Candlestick Chart Analyzer",
    page_icon="📊",
    layout="wide"
)

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
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

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
            key_dim=PROJECTION_DIM // NUM_HEADS
        )(x1, x1)
        x2 = layers.Add()([attn, encoded])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(PROJECTION_DIM * 2, activation="gelu")(x3)
        x3 = layers.Dense(PROJECTION_DIM)(x3)
        encoded = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded)
    x = layers.GlobalAveragePooling1D()(x)
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
    return keras.Model(base.input, layers.Dense(256, activation="relu")(base.output))


def build_ensemble():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    vit = build_vit()(inputs)
    eff = build_efficientnet()(inputs)

    x = layers.Concatenate()([vit, eff])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# ================================
# PREPROCESSING
# ================================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def encode_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ================================
# MODEL LOADER (ROBUST)
# ================================
@st.cache_resource
def load_model():
    custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}

    if os.path.exists("ensemble_final_model.keras"):
        try:
            return keras.models.load_model(
                "ensemble_final_model.keras",
                custom_objects=custom_objects,
                compile=False
            )
        except:
            pass

    for wf in [
        "ensemble_final_model.weights.h5",
        "ensemble_final_model_converted.h5"
    ]:
        if os.path.exists(wf):
            model = build_ensemble()
            model.load_weights(wf)
            return model

    return None

# ================================
# GPT-4o VISION (SAFE JSON)
# ================================
def analyze_with_gpt(image):
    img_b64 = encode_image(image)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical analyst. Respond ONLY in JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this candlestick chart."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1200,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}

# ================================
# UI
# ================================
st.markdown("""
<style>
.big-title {font-size:2.4rem;font-weight:700;text-align:center;}
.card {background:#111;border-radius:12px;padding:1.2rem;margin-bottom:1rem;}
.metric {font-size:1.4rem;font-weight:600;color:#4fd1c5;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 AI Candlestick Chart Analyzer</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>EfficientNet + Vision Transformer + GPT-4o</p>", unsafe_allow_html=True)
st.divider()

model = load_model()
if model is None:
    st.error("❌ Model could not be loaded.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    img_file = st.file_uploader("Upload Chart Image", type=["png", "jpg", "jpeg"])
    if img_file:
        image = Image.open(img_file)
        st.image(image, use_container_width=True)

with col2:
    if img_file and st.button("🔬 Analyze Chart", use_container_width=True):
        with st.spinner("Analyzing chart..."):
            gpt = analyze_with_gpt(image)
            pred = model.predict(preprocess_image(image), verbose=0)
            bias = CLASSES[np.argmax(pred[0])]

        if "error" in gpt:
            st.error(gpt["error"])
        else:
            st.markdown("<div class='card'><div class='metric'>Model Bias</div>"
                        f"{'Bullish' if bias=='buyinput' else 'Bearish'}</div>", unsafe_allow_html=True)
            st.json(gpt)

st.divider()
st.caption("⚠️ Educational purpose only. Not financial advice.")
