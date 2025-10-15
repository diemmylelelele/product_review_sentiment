FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system packages only if needed; slim base includes most runtime libs
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Preinstall requirements
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Ensure required folders exist at runtime
RUN mkdir -p artifacts/faiss uploads data/processed

# Expose Flask port
EXPOSE 5000

# Default to running the Flask app
CMD ["python", "app.py"]
