An educational market interpretation system combining a trained Ensemble Model (EfficientNetB0 + Vision Transformer) with GPT-4 Vision for hybrid candlestick chart analysis.
🎯 Project Overview
This system analyzes candlestick chart images using a hybrid approach:

Trained Model: EfficientNetB0 + Vision Transformer ensemble trained on 10,000 images (5K buy patterns, 5K sell patterns)
GPT-4 Vision: For sideways market detection and detailed technical analysis
Hybrid Pipeline: Combines both approaches with 60% GPT-4 weight and 40% model weight for trending markets

🏗️ Architecture
Analysis Pipeline

Initial Check (GPT-4 Vision)

Determines if market is SIDEWAYS or TRENDING
If sideways → Returns GPT-4 analysis only


Trending Market Analysis

Runs image through trained ensemble model
Gets detailed GPT-4 technical analysis
Combines results: 60% GPT + 40% Model



Model Architecture

EfficientNetB0: Transfer learning from ImageNet
Vision Transformer: Custom ViT with 6 transformer layers
Ensemble: Concatenated features from both models

📋 Requirements
System Requirements

Python 3.10+
8GB+ RAM recommended
GPU optional (model runs fine on CPU)

API Requirements

OpenAI API key with GPT-4 Vision access



