# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install wheel for better package installation
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files first for better caching
COPY requirements*.txt ./

# Install Python dependencies with fallback for conflicts
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Falling back to minimal requirements..." && \
     pip install --no-cache-dir -r requirements-minimal.txt)

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed_pdfs data/embeddings data/vector_store .streamlit

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

# Expose port (will be overridden by cloud platform)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start command with proper signal handling and PORT support
CMD streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true 