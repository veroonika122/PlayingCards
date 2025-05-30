# 62533/62T22 Applied machine learning and big data, Spring 2025
**Final handin: January 24th midnight (11.59PM)**
___
## Project Description

**Authors:**  
Veroonika Tamm (s250069)  
Stiina Salumets (s250088)

**Course:** 62533/62T22 Applied machine learning and big data, Spring 2025  
**Date:** May 30, 2024

## 1. Problem Definition
We developed an end-to-end machine learning pipeline for playing card classification. The goal was to create a system that could accurately identify playing cards from images, implementing a complete pipeline from data processing to model deployment.

## 2. Data Extraction and Data Processing
The dataset used is the [Playing Cards dataset from Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data). It consists of:
- 7624 training images
- 265 validation images
- 265 test images
- 53 classes (52 standard playing cards + 1 joker)

Data processing pipeline includes:
- Automatic data download using Kaggle API
- Image preprocessing and normalization
- Data versioning using DVC
- Tensor conversion and storage for efficient training

## 3. Modeling
We utilized PyTorch Image Models (TIMM) framework to implement our solution:
- Base model: Pre-trained ResNet18
- Fine-tuning approach: Freezing initial layers and training final layers
- Training optimization:
  - Memory-efficient data loading
  - Gradient accumulation
  - Model export to ONNX format for deployment

## 4. Performance Evaluation
The model was evaluated using multiple metrics:
- Accuracy on validation set
- Training and validation loss curves
- Real-time inference performance
- Model robustness testing

Testing framework includes:
- Unit tests for data processing
- Model architecture tests
- API endpoint testing
- Continuous Integration with GitHub Actions:
  - Automated testing on multiple Python versions
  - Code quality checks with pre-commit hooks
  - Docker build verification
  - Cloud deployment tests

## 5. Results Explanation
The project successfully achieved:
- High accuracy in card classification
- Efficient deployment pipeline using Docker and Cloud services
- Real-time inference capabilities
- Scalable and maintainable codebase

### Project Structure
```txt
├── .github/                  # CI/CD workflows
│   └── workflows/           # GitHub Actions configurations
├── configs/                  # Configuration files
│   ├── cloudbuild.yaml     # Cloud build configuration
│   ├── config_gpu.yaml     # GPU training settings
│   └── vertex_ai_train.yaml # Cloud AI platform config
├── data/                     # Data directory
│   ├── processed/           # Processed tensors
│   └── raw/                 # Raw image data
├── deployment/              # Deployment files
│   ├── api_v2.py           # ONNX inference API
│   └── frontend.py         # Web interface
├── dockerfiles/             # Docker configurations
├── models/                  # Trained models
├── src/                     # Source code
│   └── playing_cards/
│       ├── data.py         # Data processing
│       ├── model.py        # Model architecture
│       └── train.py        # Training pipeline
└── tests/                   # Unit tests
```

### Running the Project
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
export PYTHONPATH=<path-to-project>/src
pytest tests/
```

3. Train model:
```bash
python src/playing_cards/train.py
```

4. Cloud Training (optional):
```bash
# Configure cloud settings in configs/vertex_ai_train.yaml
python src/playing_cards/train_cloud.py
```

### Dependencies
- PyTorch
- TIMM (PyTorch Image Models)
- FastAPI
- ONNX Runtime
- DVC

### Development Tools
- GitHub Actions for CI/CD
- Pre-commit hooks for code quality
- Docker for containerization
- Cloud services for scalable training and deployment

### References
1. Playing Cards Dataset. Kaggle. https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data
2. PyTorch Image Models (TIMM). https://github.com/huggingface/pytorch-image-models
3. FastAPI Documentation. https://fastapi.tiangolo.com/
4. ONNX Runtime Documentation. https://onnxruntime.ai/
5. Data Version Control (DVC). https://dvc.org/
