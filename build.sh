#!/bin/bash
# build.sh - Custom build script to avoid CUDA dependencies

set -e

# Create a clean virtual environment
echo "🧹 Creating clean virtual environment..."
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Install pip and setuptools first
echo "📦 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install numpy first to avoid dependency conflicts
echo "📦 Installing numpy..."
pip install numpy==2.0.0

# Install scikit-learn (this will pull in scipy)
echo "📦 Installing scikit-learn..."
pip install scikit-learn==1.5.0

# Install xgboost with CPU-only flags
echo "📦 Installing xgboost (CPU-only)..."
pip install xgboost==2.1.4 --no-binary xgboost --no-cache-dir

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt --no-deps

# Install joblib separately (needed for model loading)
echo "📦 Installing joblib..."
pip install joblib==1.4.2

# Verify the installation
echo "✅ Verifying installation..."
pip list | grep -E "(numpy|scipy|scikit-learn|xgboost|nvidia)"

echo "🎉 Build completed successfully!"