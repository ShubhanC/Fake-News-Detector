#!/bin/bash
# deploy.sh - Deploy Fake News Detector to Vercel with data exclusion workaround

# Exit on error
echo "🚀 Starting deployment with data exclusion workaround..."
set -e

# Clean up any Docker-related files
echo "🧹 Cleaning Docker-related files..."
rm -f Dockerfile Dockerfile.social deploy-docker.sh

# Create backup directory
BACKUP_DIR="/tmp/fake-news-data-backup-$(date +%s)"
echo "📁 Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Move large data directory out of the way
echo "📂 Moving data directory out of build path..."
if [ -d "./data" ]; then
    echo "   Moving ./data to $BACKUP_DIR/"
    mv "./data" "$BACKUP_DIR/"
fi

# Clean previous build artifacts
echo "🧹 Cleaning previous build artifacts..."
rm -rf .vercel
rm -f pyproject.toml uv.lock

# Build
set +e  # Don't exit on build error - we want to restore data
echo "🛠️  Building with Vercel (prebuilt)..."
vercel build --prod --yes
BUILD_RESULT=$?
set -e

# Restore data directory
echo "🔄 Restoring data directory..."
if [ -d "$BACKUP_DIR/data" ]; then
    echo "   Moving $BACKUP_DIR/data back to ./"
    mv "$BACKUP_DIR/data" "./"
fi

# Clean up
rm -rf "$BACKUP_DIR"

# Check build result
if [ $BUILD_RESULT -ne 0 ]; then
    echo "❌ Build failed. Data directory restored."
    exit $BUILD_RESULT
fi

# Deploy with prebuilt
if [ -d ".vercel/output" ]; then
    echo "🚀 Deploying prebuilt output to Vercel..."
    vercel deploy --prebuilt --prod
else
    echo "❌ No build output found. Cannot deploy."
    exit 1
fi

# Success
echo "✅ Deployment complete!
"
echo "🔗 Production URL: https://fake-news-detector-sc.vercel.app"
echo "🔍 Inspect: https://vercel.com/shubhancs-projects/fake-news-detector"