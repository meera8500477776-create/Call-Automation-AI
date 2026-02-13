# Multi-stage build for Config-Driven Voice Agent Platform
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY .env.example .env

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN useradd -m -u 1000 voiceagent && \
    chown -R voiceagent:voiceagent /app

USER voiceagent

# Health check endpoint (assumes app exposes /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port for API/WebSocket
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "voice_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
