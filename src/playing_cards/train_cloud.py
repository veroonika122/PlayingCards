import os
import hydra
import torch
from omegaconf import DictConfig

@hydra.main(config_path="../../configs", config_name="default_config.yaml")
def train(config: DictConfig):
    """Train the model on cloud."""
    print("Training on cloud...")
    
    # Set the project directory to the cloud storage bucket
    project_dir = "/gcs/playing-cards-bucket"
    
    # Your training code here
    print(f"Using project directory: {project_dir}")
    
    # Save model to cloud storage
    if not os.path.exists(f"{project_dir}/models"):
        os.makedirs(f"{project_dir}/models")
    
    torch.save(model.state_dict(), f"{project_dir}/models/model.pt")

if __name__ == "__main__":
    train()
