import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from src.dataset import OffroadDataset, get_val_transforms
from src.model import create_model
from src.metrics import compute_iou
from src.utils import load_config, setup_directories

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--weights', type=str, required=True, help='path to saved model weights (e.g. saved_model_weights/best.pth)')
    parser.add_argument('--bottom-percent', type=int, default=30, help='percentage of training samples to designate as hard')
    return parser.parse_args()

def main():
    args = get_args()
    config = load_config(args.config)
    setup_directories(config['logging'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Mining hard examples on device: {device}")
    
    # Use validation transforms (no augmentation) for mining inference
    # to evaluate pure baseline performance on the train set
    test_transforms = get_val_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    
    # We load the TRAINING dataset directly
    train_dir_color = os.path.join(config['dataset']['train_dir'], 'Color_Images')
    train_dir_seg = os.path.join(config['dataset']['train_dir'], 'Segmentation')
    
    train_dataset = OffroadDataset(
        images_dir=train_dir_color,
        masks_dir=train_dir_seg,
        class_mapping=config['class_mapping'],
        transforms=test_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=config['training']['num_workers'])
    
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
    
    image_metrics = []
    
    # Class mapping rare IDs
    # 4: Ground Clutter, 5: Flowers, 6: Logs
    rare_classes = [4, 5, 6]
    
    print("Beginning inference over training set to identify failures...")
    with torch.no_grad():
        for images, masks, img_names in tqdm(train_loader, desc="Mining Iteration"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for b_idx in range(images.size(0)):
                b_ious, b_miou = compute_iou(preds[b_idx:b_idx+1], masks[b_idx:b_idx+1], config['model']['num_classes'])
                
                # Check rare class presence in true mask
                mask_np = masks[b_idx].cpu().numpy()
                rare_score = 0.0
                
                for rc in rare_classes:
                    pixel_count = np.sum(mask_np == rc)
                    if pixel_count > 0:
                        # Add a strong hardness bonus if the image contains a difficult class
                        rare_score += 0.3
                        
                # Hardness Score: higher means more difficult
                # Base is derived from inverse mIoU
                hardness_score = (1.0 - b_miou) + rare_score
                
                image_metrics.append({
                    "img_name": img_names[b_idx],
                    "miou": b_miou,
                    "rare_score": rare_score,
                    "hardness_score": hardness_score
                })

    # Sort descending by hardness
    image_metrics.sort(key=lambda x: x["hardness_score"], reverse=True)
    
    num_hard = int((args.bottom_percent / 100.0) * len(image_metrics))
    hard_cases = image_metrics[:num_hard]
    
    print(f"\nIdentified {num_hard} hard examples out of {len(image_metrics)} total.")
    print(f"Top 3 Hardest Examples:")
    for hc in hard_cases[:3]:
        print(f"  Img: {hc['img_name']} | mIoU: {hc['miou']:.4f} | Rare Bonus: {hc['rare_score']:.2f} | Hardness: {hc['hardness_score']:.4f}")
    
    out_dict = {
        "failure_cases": hard_cases
    }
    
    os.makedirs(config['logging']['failure_cases_dir'], exist_ok=True)
    out_path = os.path.join(config['logging']['failure_cases_dir'], 'metadata.json')
    
    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"\nHard example metadata saved to: {out_path}")

if __name__ == '__main__':
    main()
