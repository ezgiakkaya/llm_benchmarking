#!/bin/bash

# Set default port if not provided by cloud platform
export PORT=${PORT:-8501}

echo "🚀 Starting COMP430 LLM Benchmark on port $PORT"
echo "📊 Environment: ${NODE_ENV:-production}"
echo "🗄️ MongoDB URI: ${MONGODB_URI:-not-set}"

# Start the Streamlit application
exec streamlit run app/main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=false \
    --browser.gatherUsageStats=false 