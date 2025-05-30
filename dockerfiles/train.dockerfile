# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for reduced memory usage
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Create working directory
WORKDIR /app

# Copy only necessary files first
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_dev.txt

# Copy the rest of the application
COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs/ configs/
COPY models/ models/
COPY docs/ docs/
COPY tasks.py tasks.py
COPY reports/ reports/

# Install the package
RUN pip install . --no-deps --no-cache-dir

# Skip Kaggle download by copying data directly
COPY data/ data/

# Set the entrypoint with memory optimization flags
ENTRYPOINT ["python", "-X", "faulthandler", "-u", "src/project_mlops/train.py"]
