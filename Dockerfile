FROM python:3.10-slim

LABEL org.opencontainers.image.title="TCR Leakage Benchmark"
LABEL org.opencontainers.image.description="Reproducible synthetic benchmark for ML leakage in paired biomedical outcomes"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Verify imports work
RUN python -c "import numpy, pandas, sklearn, shap, xgboost; print('All imports OK')"

# Default: run the full pipeline
CMD ["bash", "run_all.sh"]
