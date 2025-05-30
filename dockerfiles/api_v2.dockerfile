FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deployment/api_v2.py app.py
COPY deployment/label_converter.npy label_converter.npy
COPY deployment/resnet18_model.onnx resnet18_model.onnx
COPY deployment/requirements_api.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 