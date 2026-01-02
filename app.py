import os
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "ensemble_final_model.keras"
IMG_SIZE = 224  # CRITICAL: Must match training size
try:
    from config import OPENAI_API_KEY
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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
# CRITICAL: These are exact copies from your training script
# They must be defined for model loading

class Patches(layers.Layer):
    """Extract patches from images - exact copy from training"""
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
    """Encode patches with position embeddings - exact copy from training"""
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
# MODEL RECONSTRUCTION FUNCTIONS
# ================================
def build_vit():
    """Rebuild Vision Transformer - EXACT COPY from training"""
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
    """Rebuild EfficientNet branch - EXACT COPY from training"""
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
    """Rebuild complete ensemble - EXACT COPY from training"""
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
# PREPROCESSING FUNCTIONS
# ================================
def preprocess_image_for_model(image):
    """
    Preprocesses uploaded image EXACTLY as training data was processed.
    
    Training preprocessing steps:
    1. Resize to 224x224 (IMG_SIZE from training)
    2. Convert to RGB (3 channels)
    3. Rescale pixel values to [0, 1] range (divided by 255)
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed numpy array ready for model inference
    """
    # Step 1: Ensure RGB format (training used RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Step 2: Resize to exact training dimensions (224x224)
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Step 3: Convert to numpy array
    img_array = np.array(image)
    
    # Step 4: Rescale to [0, 1] range (training used rescale=1.0/255)
    img_array = img_array.astype('float32') / 255.0
    
    # Step 5: Add batch dimension (model expects batch input)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def encode_image_to_base64(image):
    """
    Encodes PIL Image to base64 string for GPT-4 Vision API.
    
    Args:
        image: PIL Image object
    
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_trained_model():
    """
    Loads the trained ensemble model with custom objects.
    Compatible with TensorFlow 2.13-2.15 and Python 3.9+
    
    Method 1: Try loading saved model with custom objects
    Method 2: Try loading weights into rebuilt architecture
    Method 3: Build fresh architecture (requires retraining)
    
    Returns:
        Loaded Keras model or None
    """
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    # Method 1: Try loading full model
    if os.path.exists(MODEL_PATH):
        try:
            st.info("🔄 Loading model...")
            model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load full model: {str(e)[:100]}...")
    
    # Method 2: Try loading weights into rebuilt architecture
    weight_files = [
        'ensemble_final_model.weights.h5',        # Keras 3 format
        'ensemble_final_model_weights.h5',        # Legacy format
        'ensemble_final_model_converted.h5',      # Converted full model
        'ensemble_final_model_converted.keras',   # Converted native format
        MODEL_PATH.replace('.keras', '_weights.h5')
    ]
    
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            try:
                st.info(f"🔄 Rebuilding architecture and loading weights from {weight_file}...")
                model = build_ensemble()
                model.load_weights(weight_file)
                st.success("✅ Model loaded with weights successfully!")
                return model
            except Exception as e:
                st.warning(f"⚠️ Could not load {weight_file}: {str(e)[:100]}...")
                continue
    
    # Method 3: All failed
    st.error("❌ Could not load model.")
    st.error("**Required files (one of these):**")
    st.error("1. `ensemble_final_model.keras` (full model)")
    st.error("2. `ensemble_final_model_weights.h5` (weights only)")
    st.error("3. `ensemble_final_model_converted.h5` (converted model)")
    
    st.info("**To fix this:**")
    st.info("1. Run: `python convert_model.py`")
    st.info("2. Or copy your model file to this directory")
    st.info("3. Or retrain the model")
    
    return None

# ================================
# GPT-4 VISION API INTEGRATION
# ================================
def analyze_with_gpt4_vision(image, is_trend_check=True):
    """
    Sends image to GPT-4 Vision for analysis.
    
    Args:
        image: PIL Image object
        is_trend_check: If True, checks for sideways vs trending market
                       If False, performs detailed technical analysis
    
    Returns:
        Dict with analysis results or None if error
    """
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not found in .env file"}
    
    # Encode image
    base64_image = encode_image_to_base64(image)
    
    # Updated prompt for comprehensive analysis
    system_prompt = """You are an expert technical analyst with deep knowledge of price action, candlestick patterns, support/resistance, Fibonacci levels, and market structure. Analyze the candlestick chart image comprehensively and provide:

1. **Trend Direction**: Identify if the market is in an UPTREND, DOWNTREND, or SIDEWAYS/RANGING movement by analyzing the overall price structure.

2. **Trend Strength**: Assess if the trend is STRONG, MODERATE, or WEAK based on:
   - Momentum and slope of price movement
   - Consistency of higher highs/lower lows
   - Volume patterns (if visible)
   - Strength of impulse moves vs corrections

3. **Market Phase**: Determine the current market phase based on price action on the right side (most recent candles):
   - ACCUMULATION: Price consolidating at lower levels, potential bottom formation, decreasing volatility
   - DISTRIBUTION: Price consolidating at higher levels, potential top formation, weakening momentum
   - MARKUP: Strong bullish momentum, healthy pullbacks, buyers in control, price making higher highs
   - MARKDOWN: Strong bearish momentum, failed rallies, sellers in control, price making lower lows
   - MANIPULATION: Erratic price action, false breakouts, stop hunts, whipsaw movements, indecision

4. **Pattern Formation**: Identify candlestick patterns and chart patterns visible in the chart:
   - Candlestick Patterns: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star, Three White Soldiers/Black Crows, etc.
   - Chart Patterns: Head & Shoulders, Double/Triple Tops/Bottoms, Triangles (Ascending/Descending/Symmetrical), Flags, Pennants, Wedges, Channels
   - Support/Resistance: Key horizontal levels where price has reacted multiple times
   - Fibonacci Levels: Any visible Fibonacci retracement or extension levels (23.6%, 38.2%, 50%, 61.8%, etc.)
   - Trendlines: Ascending, descending, or horizontal trendlines
   Classify overall as: BULLISH (favoring upside), BEARISH (favoring downside), or NEUTRAL/SIDEWAYS (ranging/indecisive)

5. **Technical Indicators Analysis**: If visible in the chart, analyze:
   - Moving Averages: Price position relative to MAs, MA crossovers
   - RSI/Momentum: Overbought/oversold conditions, divergences
   - Volume: Increasing/decreasing, volume confirmation
   - MACD: Signal line crossovers, histogram direction
   - Bollinger Bands: Price at bands, squeeze conditions
   If not visible, focus purely on price action.

6. **Actionable Tips**: Provide 3-5 specific, educational tips based on the complete analysis including:
   - What to watch for (specific price levels, pattern completions)
   - Risk factors and warning signs
   - Confirmation signals to look for
   - How support/resistance and Fibonacci levels play a role
   - Time frame considerations

7. **Trading Consideration**: Based on the COMPLETE technical picture (patterns, S/R, Fibonacci, trend, phase), suggest:
   - BUY: If multiple bullish factors align (bullish patterns, support holding, uptrend intact, accumulation/markup phase)
   - SELL: If multiple bearish factors align (bearish patterns, resistance holding, downtrend intact, distribution/markdown phase)
   - WAIT: If signals are mixed, pattern incomplete, at major S/R, or in manipulation phase

Respond in JSON format:
{
  "trend_direction": "uptrend/downtrend/sideways",
  "trend_strength": "strong/moderate/weak",
  "market_phase": "accumulation/distribution/markup/markdown/manipulation",
  "market_phase_explanation": "detailed explanation with reasoning",
  "pattern_formation": "bullish/bearish/neutral",
  "pattern_description": "detailed description including candlestick patterns, chart patterns, S/R levels, Fibonacci levels if visible, and overall structure",
  "actionable_tips": ["tip1 with specific levels/patterns", "tip2", "tip3", "tip4", "tip5"],
  "trading_consideration": "buy/sell/wait",
  "consideration_reasoning": "comprehensive reasoning mentioning all technical factors: patterns, support/resistance, Fibonacci levels, trend alignment, market phase, and risk factors"
}

CRITICAL RULES:
- Analyze the ENTIRE chart, not just recent candles
- Identify ALL visible support and resistance levels
- Look for Fibonacci retracement/extension levels if visible
- Identify specific candlestick patterns (name them)
- Consider pattern completion and confirmation
- Multi-timeframe perspective where possible
- Be specific with technical terminology
- Provide educational, detailed analysis
- Consider the interplay of ALL technical factors before final recommendation"""
    
    # API request payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this candlestick chart image comprehensively, focusing on current price action on the right side."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Try to parse JSON response
        try:
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            return analysis
        except json.JSONDecodeError:
            return {"error": "Failed to parse GPT-4 response", "raw_content": content}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# ================================
# HYBRID ANALYSIS PIPELINE
# ================================
def perform_hybrid_analysis(image, model):
    """
    Main analysis pipeline combining GPT-4 Vision and trained model.
    
    Args:
        image: PIL Image object
        model: Loaded Keras model
    
    Returns:
        Dict with complete analysis results
    """
    results = {}
    
    # Step 1: Get comprehensive GPT-4 analysis
    with st.spinner("🔍 Analyzing chart with AI vision..."):
        gpt_analysis = analyze_with_gpt4_vision(image, is_trend_check=False)
    
    if "error" in gpt_analysis:
        return {"error": gpt_analysis["error"]}
    
    # Step 2: Get model prediction for additional insight
    with st.spinner("🤖 Running ensemble model..."):
        preprocessed = preprocess_image_for_model(image)
        prediction = model.predict(preprocessed, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        model_confidence = float(prediction[0][predicted_class_idx]) * 100
        
        model_label = CLASSES[predicted_class_idx]
        model_bias = "Bullish" if model_label == "buyinput" else "Bearish"
    
    # Combine results
    results["trend_direction"] = gpt_analysis.get("trend_direction", "sideways").capitalize()
    results["trend_strength"] = gpt_analysis.get("trend_strength", "moderate").capitalize()
    results["market_phase"] = gpt_analysis.get("market_phase", "N/A").capitalize()
    results["market_phase_explanation"] = gpt_analysis.get("market_phase_explanation", "Market phase analysis not available")
    results["pattern_formation"] = gpt_analysis.get("pattern_formation", "neutral").capitalize()
    results["pattern_description"] = gpt_analysis.get("pattern_description", "Pattern analysis not available")
    results["actionable_tips"] = gpt_analysis.get("actionable_tips", [
        "Monitor price action for confirmation",
        "Watch for volume changes",
        "Consider risk management strategies"
    ])
    results["trading_consideration"] = gpt_analysis.get("trading_consideration", "wait").upper()
    results["consideration_reasoning"] = gpt_analysis.get("consideration_reasoning", "Further confirmation needed")
    results["model_bias"] = model_bias
    
    return results

# ================================
# STREAMLIT UI
# ================================
def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Chart Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.95;
        }
        .analysis-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #667eea;
        }
        .card-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #667eea;
            margin: 0.5rem 0;
        }
        .description-text {
            color: #4a5568;
            line-height: 1.6;
            margin-top: 0.5rem;
        }
        .tips-list {
            background: #f7fafc;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 0.5rem;
        }
        .tip-item {
            padding: 0.5rem 0;
            color: #2d3748;
            line-height: 1.5;
        }
        .trading-action {
            background: #edf2f7;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1.5rem 0;
        }
        .action-label {
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 0.5rem;
        }
        .action-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        .action-buy { color: #48bb78; }
        .action-sell { color: #f56565; }
        .action-wait { color: #ed8936; }
        .disclaimer-box {
            background: #fff5f5;
            border: 2px solid #fc8181;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 2rem 0;
        }
        .disclaimer-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #c53030;
            margin-bottom: 0.5rem;
        }
        .disclaimer-text {
            color: #742a2a;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <div class="main-title">📊 AI Candlestick Chart Analyzer</div>
            <div class="subtitle">Advanced Technical Analysis with EfficientNet + Vision Transformer + GPT-4</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.stop()
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Chart")
        uploaded_file = st.file_uploader(
            "Upload a candlestick chart image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a clear candlestick chart screenshot"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chart", use_container_width=True)
            
            if st.button("🔬 Analyze Chart", type="primary", use_container_width=True):
                if not OPENAI_API_KEY:
                    st.error("❌ OpenAI API key not found. Please add it to your .env file.")
                    st.stop()
                
                # Perform analysis
                results = perform_hybrid_analysis(image, model)
                
                # Store in session state
                st.session_state['results'] = results
    
    with col2:
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            if "error" in results:
                st.error(f"❌ Analysis Error: {results['error']}")
            else:
                st.markdown("### 📈 Analysis Results")
                
                # 1. Trend Direction
                st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-header">📊 Trend Direction</div>
                        <div class="metric-value">{results['trend_direction']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 2. Trend Strength
                st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-header">💪 Trend Strength</div>
                        <div class="metric-value">{results['trend_strength']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 3. Market Phase
                st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-header">🔄 Market Phase</div>
                        <div class="metric-value">{results['market_phase']}</div>
                        <div class="description-text">{results['market_phase_explanation']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 4. Pattern Formation
                st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-header">📐 Pattern Formation</div>
                        <div class="metric-value">{results['pattern_formation']} Pattern</div>
                        <div class="description-text">{results['pattern_description']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 5. Actionable Tips
                tips_html = '<div class="tips-list">'
                for i, tip in enumerate(results['actionable_tips'], 1):
                    tips_html += f'<div class="tip-item"><strong>{i}.</strong> {tip}</div>'
                tips_html += '</div>'
                
                st.markdown(f"""
                    <div class="analysis-card">
                        <div class="card-header">💡 Actionable Tips</div>
                        {tips_html}
                    </div>
                """, unsafe_allow_html=True)
                
                # 6. Trading Consideration
                action = results['trading_consideration']
                action_class = f"action-{action.lower()}"
                
                st.markdown(f"""
                    <div class="trading-action">
                        <div class="action-label">Suggested Action</div>
                        <div class="action-value {action_class}">{action}</div>
                        <div class="description-text" style="margin-top: 1rem;">{results['consideration_reasoning']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Disclaimer
                st.markdown("""
                    <div class="disclaimer-box">
                        <div class="disclaimer-title">⚠️ IMPORTANT DISCLAIMER</div>
                        <div class="disclaimer-text">
                            This analysis is provided for <strong>educational and informational purposes only</strong>. 
                            It does NOT constitute financial advice, investment recommendations, or trading signals. 
                            The suggested action (Buy/Sell/Wait) is based on technical analysis patterns and should NOT 
                            be used as the sole basis for making trading decisions.
                            <br><br>
                            <strong>Trading involves substantial risk of loss.</strong> Always conduct your own research, 
                            consider multiple factors, use proper risk management, and consult with licensed financial 
                            advisors before making any investment decisions. Past patterns do not guarantee future results.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #718096; padding: 1rem;">
            <p style="margin: 0;">Powered by EfficientNetB0 + Vision Transformer Ensemble & GPT-4o Mini</p>
            <p style="margin: 0; font-size: 0.9rem;">Academic Project | Educational Purpose Only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()