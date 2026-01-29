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
CLASSES = ["buyinput", "sellinput"]
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 512
NUM_HEADS = 8
TRANSFORMER_LAYERS = 6

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with error handling"""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            st.error("⚠️ OpenAI API key not found. Please add it to Streamlit secrets.")
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

# ================================
# CUSTOM LAYERS
# ================================
class Patches(layers.Layer):
    """Extract patches from images"""
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
    """Encode patches with position embeddings"""
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
        cfg = super().get_config()
        cfg.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return cfg

# ================================
# MODEL BUILDING
# ================================
@st.cache_resource
def build_vit():
    """Build Vision Transformer model"""
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

@st.cache_resource
def build_efficientnet():
    """Build EfficientNet model"""
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    for layer in base.layers[:-20]:
        layer.trainable = False
    return keras.Model(base.input, layers.Dense(256, activation="relu")(base.output))

@st.cache_resource
def build_ensemble():
    """Build ensemble model combining ViT and EfficientNet"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    vit = build_vit()(inputs)
    eff = build_efficientnet()(inputs)
    x = layers.Concatenate()([vit, eff])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# ================================
# MODEL LOADER (ROBUST)
# ================================
@st.cache_resource
def load_model():
    """Load trained model with multiple fallback strategies"""
    custom_objects = {
        "Patches": Patches,
        "PatchEncoder": PatchEncoder
    }
    
    # Try loading .keras file
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
            st.success("✅ Model loaded successfully from .keras file")
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load .keras file: {e}")
    
    # Try loading weights files
    weight_files = [
        "ensemble_final_model.weights.h5",
        "ensemble_final_model_converted.h5",
        "model.weights.h5"
    ]
    
    for wf in weight_files:
        if os.path.exists(wf):
            try:
                model = build_ensemble()
                model.load_weights(wf)
                st.success(f"✅ Model weights loaded from {wf}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Could not load weights from {wf}: {e}")
    
    # If no model found, build fresh model (untrained)
    st.warning("⚠️ No trained model found. Building untrained model architecture...")
    st.info("📝 To use a trained model, upload your model file (.keras or .h5) to the app directory")
    return build_ensemble()

# ================================
# PREPROCESSING
# ================================
def preprocess_image(image):
    """Preprocess image for model input"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def encode_image(image):
    """Encode image to base64"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ================================
# GPT-4o VISION ANALYSIS
# ================================
def analyze_with_gpt(image, client):
    """Analyze chart using GPT-4o Vision with robust error handling"""
    if client is None:
        return {"error": "OpenAI client not initialized"}
    
    img_b64 = encode_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a technical analyst expert. Analyze the candlestick chart and provide insights.
                    Respond ONLY in valid JSON format with these keys:
                    - trend: overall trend (bullish/bearish/neutral)
                    - patterns: list of identified patterns
                    - support_resistance: key levels
                    - recommendation: trading recommendation
                    - confidence: your confidence level (0-100)
                    - analysis: brief textual analysis"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this candlestick chart and provide technical analysis."
                        },
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
        
        # Clean JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON response from GPT",
            "raw_response": content if 'content' in locals() else "No response"
        }
    except Exception as e:
        return {"error": str(e)}

# ================================
# UI STYLING
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .bullish {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .bearish {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
st.markdown('<div class="main-header">📊 AI Candlestick Chart Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">EfficientNet + Vision Transformer + GPT-4o</div>', unsafe_allow_html=True)
st.divider()

# Initialize
client = get_openai_client()
model = load_model()

if model is None:
    st.error("❌ Critical error: Model could not be initialized.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Chart")
    img_file = st.file_uploader(
        "Upload a candlestick chart image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear candlestick chart for analysis"
    )
    
    if img_file:
        image = Image.open(img_file)
        st.image(image, use_container_width=True, caption="Uploaded Chart")

with col2:
    st.markdown("### 🔬 Analysis")
    
    if img_file:
        if st.button("🚀 Analyze Chart", use_container_width=True):
            
            with st.spinner("🔄 Analyzing chart with AI models..."):
                
                # Model prediction
                try:
                    pred = model.predict(preprocess_image(image), verbose=0)
                    bias = CLASSES[np.argmax(pred[0])]
                    confidence = float(np.max(pred[0]) * 100)
                    
                    # Display model prediction
                    st.markdown("#### 🤖 Model Prediction")
                    bias_class = "bullish" if bias == "buyinput" else "bearish"
                    bias_text = "🟢 BULLISH" if bias == "buyinput" else "🔴 BEARISH"
                    
                    st.markdown(
                        f'<div class="prediction-box {bias_class}">{bias_text}<br/>'
                        f'<small style="font-size: 0.8rem;">Confidence: {confidence:.1f}%</small></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
                
                # GPT-4o analysis
                if client:
                    with st.spinner("🧠 Getting GPT-4o analysis..."):
                        gpt_result = analyze_with_gpt(image, client)
                        
                        if "error" in gpt_result:
                            st.error(f"❌ GPT Analysis Error: {gpt_result['error']}")
                            if "raw_response" in gpt_result:
                                with st.expander("Show raw response"):
                                    st.text(gpt_result["raw_response"])
                        else:
                            st.markdown("#### 🧠 GPT-4o Analysis")
                            st.json(gpt_result)
                else:
                    st.warning("⚠️ GPT-4o analysis unavailable (API key not configured)")
    else:
        st.info("👆 Upload a chart image to begin analysis")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Not financial advice. Always do your own research.</p>
    <p style='font-size: 0.9rem;'>Powered by TensorFlow, EfficientNet, Vision Transformers & GPT-4o</p>
</div>
""", unsafe_allow_html=True)
