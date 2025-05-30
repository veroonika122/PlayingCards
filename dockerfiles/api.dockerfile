# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

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
COPY models/ models/

# Install the package
RUN pip install . --no-deps --no-cache-dir

# Set the command
CMD ["uvicorn", "src.playing_cards.api:app", "--host", "0.0.0.0", "--port", "8000"]
