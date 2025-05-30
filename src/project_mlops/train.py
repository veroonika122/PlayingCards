import logging
import os
import sys
import gc
from datetime import datetime
from pathlib import Path
import glob

import hydra
import torch
from data import playing_cards, preprocess_data
from model import model_list
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Replace underscores with dashes in CLI arguments
sys.argv = [arg.replace("_", "-") if "--" in arg else arg for arg in sys.argv]

# Force garbage collection
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_name="hyperparams.yaml", config_path=f"{os.getcwd()}/configs", version_base="1.1")
def train(cfg) -> None:
    """Train a model on playing cards."""
    # var
    model_type = cfg.model
    batch_size = cfg.batch_size
    lr = cfg.lr if cfg.lr != None else cfg.default[model_type].lr
    epochs = cfg.epochs
    seed = cfg.seed
    base_dir = hydra.utils.get_original_cwd()  # Get the original working directory

    log = logging.getLogger(__name__)
    log.info(f"{model_type=}, {batch_size=}, {lr=}, {epochs=}, {seed=} {base_dir=}")

    # First check if we need to preprocess the data
    processed_dir = os.path.join(base_dir, "data/processed/cards-dataset")
    if not os.path.exists(processed_dir) or len(glob.glob(processed_dir + "/*.*")) == 0:
        preprocess_data()

    # model/data
    log.info(f"Using Device: {DEVICE}")
    torch.manual_seed(seed)
    model, pred_func = model_list(model_type)
    model = model.to(DEVICE)

    # Enable memory efficient loading
    torch.set_num_threads(1)
    train_set, valid_set, _ = playing_cards(base_dir)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False  # Disable pinned memory
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create models directory if it doesn't exist
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # train
    statistics = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}
    best_valid_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        samples_seen = 0

        # Training loop with memory-efficient processing
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = pred_func(model(img))
            loss = loss_fn(y_pred, target)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Update running statistics
            running_loss += loss.item() * cfg.gradient_accumulation_steps
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            running_acc += accuracy * len(target)
            samples_seen += len(target)

            # Clear some memory
            del img, target, y_pred, loss
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / (i + 1)
                avg_acc = running_acc / samples_seen
                log.info(
                    f"Epoch {epoch:>2}, iter {i + 1:>4}, train-loss: {avg_loss:.4f}, train-accuracy: {avg_acc * 100:.2f}%"
                )

                # Validation step
                model.eval()
                valid_loss = 0.0
                valid_acc = 0.0
                valid_samples = 0
                
                with torch.no_grad():
                    for val_img, val_target in valid_dataloader:
                        val_img, val_target = val_img.to(DEVICE), val_target.to(DEVICE)
                        val_pred = pred_func(model(val_img))
                        val_loss = loss_fn(val_pred, val_target)
                        
                        valid_loss += val_loss.item()
                        valid_acc += (val_pred.argmax(dim=1) == val_target).float().sum().item()
                        valid_samples += len(val_target)
                        
                        del val_img, val_target, val_pred
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                avg_valid_loss = valid_loss / len(valid_dataloader)
                avg_valid_acc = valid_acc / valid_samples
                
                log.info(
                    f"Validation - loss: {avg_valid_loss:.4f}, accuracy: {avg_valid_acc * 100:.2f}%"
                )

                # Save best model
                if avg_valid_acc > best_valid_accuracy:
                    best_valid_accuracy = avg_valid_acc
                    best_model_state = model.state_dict().copy()

                model.train()

        log.info(f"Epoch {epoch} completed")
        log.info(f"Best validation accuracy so far: {best_valid_accuracy * 100:.2f}%")

    log.info("Training completed")
    
    # Save the best model
    if best_model_state is not None:
        current_datetime = datetime.now()
        prefix = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(models_dir, f"model_{prefix}.pth")
        torch.save(best_model_state, model_path)
        log.info(f"Best model saved to: {model_path}")

        # Load best model for ONNX export
        model.load_state_dict(best_model_state)
        onnx_path = os.path.join(base_dir, "deployment", "resnet18_model.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        torch.onnx.export(model, 
                         dummy_input, 
                         onnx_path,
                         verbose=True,
                         input_names=['input'],
                         output_names=['output'])
        log.info(f"Best model exported to ONNX format at: {onnx_path}")

        # Also save the label converter
        import numpy as np
        label_converter = torch.load(f"{base_dir}/data/processed/cards-dataset/label_converter.pt", weights_only=True)
        label_converter_path = os.path.join(base_dir, "deployment", "label_converter.npy")
        np.save(label_converter_path, label_converter)
        log.info(f"Label converter saved to {label_converter_path}")

if __name__ == "__main__":
    train()
