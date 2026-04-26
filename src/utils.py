import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Define distinct colors for the 10 classes
COLOR_MAP = np.array([
    [0, 128, 0],      # Trees (Green)
    [34, 139, 34],    # Lush Bushes (Forest Green)
    [154, 205, 50],   # Dry Grass (Yellow Green)
    [184, 134, 11],   # Dry Bushes (Dark Goldenrod)
    [210, 180, 140],  # Ground Clutter (Tan)
    [255, 105, 180],  # Flowers (Hot Pink)
    [139, 69, 19],    # Logs (Saddle Brown)
    [105, 105, 105],  # Rocks (Dim Gray)
    [244, 164, 96],   # Landscape (Sandy Brown)
    [135, 206, 235],  # Sky (Sky Blue)
], dtype=np.uint8)

def setup_directories(config_run):
    dirs_to_make = [
        config_run.get('run_dir', 'runs'),
        config_run.get('train_dir', 'runs/train'),
        config_run.get('val_dir', 'runs/val'),
        config_run.get('test_dir', 'runs/test'),
        config_run.get('visualizations_dir', 'runs/visualizations'),
        config_run.get('failure_cases_dir', 'runs/failure_cases'),
        "saved_model_weights"
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

def visualize_prediction(image, mask_pred, mask_gt, save_path, img_name, iou=None):
    """
    image: normalized tensor (C, H, W)
    mask_pred: predicted tensor (H, W)
    mask_gt: ground truth tensor (H, W)
    """
    # Unnormalize image
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    mask_pred_np = mask_pred.cpu().numpy()
    color_pred = COLOR_MAP[mask_pred_np]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title(f'Original Image\n{img_name}')
    axs[0].axis('off')
    
    axs[1].imshow(color_pred)
    axs[1].set_title(f'Prediction\nIoU: {iou:.4f}' if iou is not None else 'Prediction')
    axs[1].axis('off')
    
    if mask_gt is not None:
        mask_gt_np = mask_gt.cpu().numpy()
        color_gt = COLOR_MAP[mask_gt_np]
        axs[2].imshow(color_gt)
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')
    else:
        axs[2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
