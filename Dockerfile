# Use an official lightweight Python base image
FROM python:3.11-slim

# Set environment variables to optimize Python runtime in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (curl for healthchecks, git/build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code into the container
COPY app/ ./app/
COPY .env .env

# Expose FastAPI application port
EXPOSE 8000

# Healthcheck endpoint verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Command to launch the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]