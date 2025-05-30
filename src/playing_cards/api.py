import io
import logging
import torch
import timm
import torchvision.transforms as transforms
from contextlib import asynccontextmanager
from http import HTTPStatus
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the labels for playing cards
LABELS = ['ace of clubs', 'two of clubs', 'three of clubs', 'four of clubs', 'five of clubs',
          'six of clubs', 'seven of clubs', 'eight of clubs', 'nine of clubs', 'ten of clubs',
          'jack of clubs', 'queen of clubs', 'king of clubs',
          'ace of diamonds', 'two of diamonds', 'three of diamonds', 'four of diamonds', 'five of diamonds',
          'six of diamonds', 'seven of diamonds', 'eight of diamonds', 'nine of diamonds', 'ten of diamonds',
          'jack of diamonds', 'queen of diamonds', 'king of diamonds',
          'ace of hearts', 'two of hearts', 'three of hearts', 'four of hearts', 'five of hearts',
          'six of hearts', 'seven of hearts', 'eight of hearts', 'nine of hearts', 'ten of hearts',
          'jack of hearts', 'queen of hearts', 'king of hearts',
          'ace of spades', 'two of spades', 'three of spades', 'four of spades', 'five of spades',
          'six of spades', 'seven of spades', 'eight of spades', 'nine of spades', 'ten of spades',
          'jack of spades', 'queen of spades', 'king of spades',
          'joker']

def create_model(num_classes=53):
    """Create a TIMM ResNet18 model with frozen early layers."""
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    # Freeze earlier layers (layer1 and layer2)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    # Fine-tune layer3, layer4, and fc layer
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Change the number of output classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model
    try:
        # Load TIMM ResNet18 model
        logger.info("Loading TIMM ResNet18 model")
        model = create_model(len(LABELS))
        model.eval()
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        # Clean up
        del model

app = FastAPI(lifespan=lifespan)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        logger.info(f"Received file: {file.filename}")
        byte_img = await file.read()
        
        # Open image
        img = Image.open(io.BytesIO(byte_img))
        logger.info(f"Original image mode: {img.mode}")
        
        # Convert to RGB (this handles P mode correctly)
        img = img.convert('RGB')
        logger.info(f"Converted to RGB")
        
        # Preprocess image
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        logger.info(f"Image preprocessed. Shape: {img_tensor.shape}")
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(output[0]).item()
            prediction = [LABELS[predicted_idx]]
            
            # Create a dictionary of probabilities for all cards
            prob_dict = {LABELS[i]: prob.item() for i, prob in enumerate(probabilities)}
        
        logger.info(f"Prediction complete. Label: {prediction}")
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probabilities": prob_dict
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
