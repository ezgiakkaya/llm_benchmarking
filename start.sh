#!/bin/bash

# Set default port if not provided by cloud platform
export PORT=${PORT:-8501}

echo "🚀 Starting COMP430 LLM Benchmark on port $PORT"
echo "📊 Environment: ${NODE_ENV:-production}"
echo "🗄️ MongoDB URI: ${MONGODB_URI:-not-set}"
echo "🌐 Connection mode: HTTP polling (WebSocket disabled for cloud compatibility)"
echo "⚙️  Config: Using production configuration"

# Use production config if in production, otherwise use default
if [ "${NODE_ENV:-production}" = "production" ]; then
    CONFIG_FILE=".streamlit/config-production.toml"
    echo "📁 Using config: $CONFIG_FILE"
else
    CONFIG_FILE=".streamlit/config.toml"
    echo "📁 Using config: $CONFIG_FILE"
fi

# Start the Streamlit application with production-optimized settings
exec streamlit run app/main.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=false \
    --server.allowRunOnSave=false \
    --server.fileWatcherType=none \
    --server.enableCORS=true \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false \
    --global.developmentMode=false 