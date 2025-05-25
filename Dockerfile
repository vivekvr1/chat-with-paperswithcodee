# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose Streamlit port
#EXPOSE 8501
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Create entrypoint script for better environment handling
USER root
RUN echo '#!/bin/bash\n\
echo "🔍 Environment Check:"\n\
echo "OPENAI_API_KEY: $(echo $OPENAI_API_KEY | cut -c1-10)...$(echo $OPENAI_API_KEY | tail -c5)"\n\
echo "UPSTASH_VECTOR_REST_URL: $UPSTASH_VECTOR_REST_URL"\n\
echo "UPSTASH_VECTOR_REST_TOKEN: $(echo $UPSTASH_VECTOR_REST_TOKEN | cut -c1-10)...$(echo $UPSTASH_VECTOR_REST_TOKEN | tail -c5)"\n\
echo "Starting Streamlit..."\n\
exec streamlit run bapp.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

USER streamlit

# Use entrypoint script
CMD ["/app/entrypoint.sh"]
