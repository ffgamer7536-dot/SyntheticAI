import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import shutil

from src.dataset import OffroadDataset, get_val_transforms
from src.model import create_model
from src.metrics import compute_iou
from src.utils import load_config, setup_directories, visualize_prediction

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--weights', type=str, required=True, help='path to saved model weights (e.g. saved_model_weights/best.pth)')
    return parser.parse_args()

def main():
    args = get_args()
    config = load_config(args.config)
    setup_directories(config['logging'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    test_transforms = get_val_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    
    test_dataset = OffroadDataset(
        os.path.join(config['dataset']['test_dir'], 'Color_Images'),
        os.path.join(config['dataset']['test_dir'], 'Segmentation'),
        config['class_mapping'],
        transforms=test_transforms
    )
    
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=config['training']['num_workers'])
    
    model = create_model(
        arch=config['model']['architecture'],
        backbone=config['model']['backbone'],
        weights=None,
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    print(f"Loading weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    
    all_ious = []
    class_ious = {i: [] for i in range(config['model']['num_classes'])}
    image_metrics = [] # To store (img_name, miou) to find failure cases
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, masks, img_names in tqdm(test_loader, desc="Test Inference"):
            images, masks = images.to(device), masks.to(device)
            
            if config.get('training', {}).get('use_tta', False):
                outputs1 = model(images)
                
                images_hf = torch.flip(images, dims=[3])
                outputs2_flipped = model(images_hf)
                outputs2 = torch.flip(outputs2_flipped, dims=[3])
                
                prob1 = torch.softmax(outputs1, dim=1)
                prob2 = torch.softmax(outputs2, dim=1)
                outputs = (prob1 + prob2) / 2.0
            else:
                outputs = model(images)
                
            preds = torch.argmax(outputs, dim=1)
            
            b_ious, b_miou = compute_iou(preds, masks, config['model']['num_classes'])
            all_ious.append(b_miou)
            
            for cls_idx, ciou in enumerate(b_ious):
                if not np.isnan(ciou):
                    class_ious[cls_idx].append(ciou)
            
            img_name = img_names[0]
            image_metrics.append((img_name, b_miou))
            
            # Save visual overlay
            vis_path = os.path.join(config['logging']['visualizations_dir'], img_name)
            visualize_prediction(images[0], preds[0], masks[0], vis_path, img_name, iou=b_miou)
            
    test_miou = np.nanmean(all_ious)
    print("="*30)
    print(f"FINAL TEST mIoU: {test_miou:.4f}")
    print("Per-class IoU:")
    for cls_idx, name in enumerate(config['classes']):
        cls_avg = np.nanmean(class_ious[cls_idx]) if class_ious[cls_idx] else np.nan
        print(f"  - {name}: {cls_avg:.4f}")
    print("="*30)
    
    with open("final_metrics.txt", "w") as f:
        f.write(f"FINAL TEST mIoU: {test_miou:.4f}\n")
        for cls_idx, name in enumerate(config['classes']):
            cls_avg = np.nanmean(class_ious[cls_idx]) if class_ious[cls_idx] else np.nan
            f.write(f"{name}: {cls_avg:.4f}\n")
            
    # Save Failure cases (lowest 20 images by IoU)
    print("Extracting highest failure cases...")
    failed_sorted = sorted(image_metrics, key=lambda x: x[1])
    failure_cases_dir = config['logging']['failure_cases_dir']
    
    for i in range(min(20, len(failed_sorted))):
        img_name, iou = failed_sorted[i]
        src_path = os.path.join(config['logging']['visualizations_dir'], img_name)
        dst_path = os.path.join(failure_cases_dir, f"fail_iou_{iou:.4f}_{img_name}")
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            
    print(f"Lowest {min(20, len(failed_sorted))} IoU predictions saved to {failure_cases_dir} for analysis.")

if __name__ == '__main__':
    main()
