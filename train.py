import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

from src.dataset import OffroadDataset, get_train_transforms, get_val_transforms, get_hard_transforms, compute_dataset_statistics
from src.model import create_model
from src.metrics import HybridLoss, compute_iou
from src.utils import load_config, setup_directories

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--hard-mining', type=str, default=None, help='path to failure_cases/metadata.json')
    parser.add_argument('--k', type=float, default=3.0, help='Weight multiplier for hard example samples')
    parser.add_argument('--fine-tune', action='store_true', help='Adjust LR for fine-tuning loops')
    return parser.parse_args()

def main():
    args = get_args()
    config = load_config(args.config)
    setup_directories(config['logging'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Setup data
    train_transforms = get_train_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    val_transforms = get_val_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
    
    train_dataset = OffroadDataset(
        os.path.join(config['dataset']['train_dir'], 'Color_Images'),
        os.path.join(config['dataset']['train_dir'], 'Segmentation'),
        config['class_mapping'],
        transforms=train_transforms
    )
    
    val_dataset = OffroadDataset(
        os.path.join(config['dataset']['val_dir'], 'Color_Images'),
        os.path.join(config['dataset']['val_dir'], 'Segmentation'),
        config['class_mapping'],
        transforms=val_transforms
    )
    
    # Compute class and image weights before loading
    class_weights, image_weights = compute_dataset_statistics(
        train_dataset, 
        config['model']['num_classes'],
        save_path=config['dataset'].get('dataset_stats_file', 'dataset_stats.json')
    )
    class_weights = class_weights.to(device)
    
    # Process Hard Example Mining
    hard_mining_set = set()
    if args.hard_mining and os.path.exists(args.hard_mining):
        print(f"Applying Hard Example Mining from {args.hard_mining} with k={args.k}")
        with open(args.hard_mining, 'r') as f:
            failures = json.load(f)['failure_cases']
        hard_mining_set = {item['img_name'] for item in failures}
        
        # Link hard transforms and constraints to dataset
        train_dataset.hard_mining_set = hard_mining_set
        train_dataset.hard_transforms = get_hard_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
        
        avg_weight = np.mean(image_weights)
        max_allowed_weight = avg_weight * 4.0  # Cap at 4x to prevent overfitting
        
        # Multiply dataloader sampler weights for hard examples
        for idx in range(len(train_dataset)):
            _, _, img_name = train_dataset[idx]
            if img_name in hard_mining_set:
                new_weight = image_weights[idx] * args.k
                image_weights[idx] = min(new_weight, max_allowed_weight)
                
    sampler = WeightedRandomSampler(
        weights=image_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              sampler=sampler, num_workers=config['training']['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                            shuffle=False, num_workers=config['training']['num_workers'])
    
    # Model
    model = create_model(
        arch=config['model']['architecture'],
        backbone=config['model']['backbone'],
        weights=config['model']['weights'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Loss, Optimizer, and Scaler for Mixed Precision
    criterion = HybridLoss(class_weights=class_weights).to(device)
    
    lr = config['training']['learning_rate']
    if args.fine_tune:
        lr = lr / 10.0
        print(f"Fine-tuning mode enabled. Adjusting base LR to {lr}")
        
    optimizer = optim.AdamW(model.parameters(), 
                            lr=lr, 
                            weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler(device='cuda')
    
    # Logging histories
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    best_val_miou = 0.0
    
    # Training Loop
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_class_ious = {i: [] for i in range(config['model']['num_classes'])}
        all_val_mious = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks, _ in pbar_val:
                images, masks = images.to(device), masks.to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                
                for b in range(images.size(0)):
                    b_ious, m_iou = compute_iou(preds[b:b+1], masks[b:b+1], config['model']['num_classes'])
                    all_val_mious.append(m_iou)
                    for cls_idx, ciou in enumerate(b_ious):
                        if not np.isnan(ciou):
                            all_class_ious[cls_idx].append(ciou)
                            
        val_loss /= len(val_loader)
        epoch_val_miou = np.nanmean(all_val_mious)
        
        history['val_loss'].append(val_loss)
        history['val_miou'].append(epoch_val_miou)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {epoch_val_miou:.4f}")
        
        # Specifically output monitoring for Rare Classes along with general breakdown
        rare_names = ['Ground Clutter', 'Flowers', 'Logs']
        print("Per-class Val IoU Dashboard:")
        for cls_idx, name in enumerate(config['classes']):
            cls_avg = np.nanmean(all_class_ious[cls_idx]) if all_class_ious[cls_idx] else np.nan
            if name in rare_names:
                print(f"  --> [RARE] {name}: {cls_avg:.4f}")
            else:
                print(f"  - {name}: {cls_avg:.4f}")
        
        # Save best model
        if epoch_val_miou > best_val_miou:
            print(f"--> Valid mIoU improved from {best_val_miou:.4f} to {epoch_val_miou:.4f}. Saving model.")
            best_val_miou = epoch_val_miou
            save_path = os.path.join(config['logging']['checkpoint_dir'], 'best.pth')
            torch.save(model.state_dict(), save_path)
            
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config['logging']['run_dir'], 'loss_curve.png'))
    plt.close()

if __name__ == '__main__':
    main()
