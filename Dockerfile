# Use Python 3.11.13 specifically
FROM python:3.11.13-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies with comprehensive requirements
RUN pip install --no-cache-dir -r requirements-docker.txt || \
    (echo "Docker requirements failed, trying main requirements..." && \
     pip install --no-cache-dir -r requirements.txt) || \
    (echo "Main requirements failed, trying minimal requirements..." && \
     pip install --no-cache-dir -r requirements-minimal.txt)

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Create necessary directories
RUN mkdir -p data/processed_pdfs data/embeddings data/vector_store .streamlit

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit

# Health check (using dynamic port)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Expose port (will be overridden by cloud platform)
EXPOSE 8501

# Run the application using startup script
CMD ["./start.sh"] 