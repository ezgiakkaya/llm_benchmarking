#!/bin/bash

# Set default port if not provided by cloud platform
export PORT=${PORT:-8501}

echo "🚀 Starting COMP430 LLM Benchmark on port $PORT"
echo "📊 Environment: ${NODE_ENV:-production}"
echo "🗄️ MongoDB URI: ${MONGODB_URI:-not-set}"
echo "🌐 WebSocket support: enabled"

# Start the Streamlit application with cloud-optimized settings
exec streamlit run app/main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=false \
    --server.enableCORS=true \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false 