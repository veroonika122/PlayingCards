# Base image
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs/ configs/
COPY docs/ docs/
COPY tasks.py tasks.py

WORKDIR /

# Install PyTorch and torchvision (CPU version)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Set environment variables
ENV PYTHONPATH=/src

# Expose the port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "src.project_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
