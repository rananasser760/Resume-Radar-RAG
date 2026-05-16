# Use Python 3.11 slim to minimize image size
FROM python:3.11-slim

# Set working environment
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Bypass the '600MB Wall'
# We install the lightweight CPU-only version of Torch independently
RUN pip install --upgrade pip --default-timeout=1000 && \
    pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install Requirements
COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt

# Final App Assembly
# Copies the 'app' folder into '/app/app' to maintain package structure
COPY app/ ./app/
COPY raw_data/ ./raw_data/
COPY .env .

EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]