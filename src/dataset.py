import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json

class OffroadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_mapping, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_mapping = class_mapping
        
        self.images = sorted(os.listdir(images_dir))
        # Ensure only images are loaded
        self.images = [f for f in self.images if f.endswith('.png') or f.endswith('.jpg')]
        self.transforms = transforms
        
        # Hard Example Mining overrides
        self.hard_mining_set = set()
        self.hard_transforms = None
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        # Since class IDs go up to 10000, we read with IMREAD_UNCHANGED to keep 16-bit data intact if applicable
        mask_path = os.path.join(self.masks_dir, img_name)
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('.jpg', '.png')
            
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # If mask is missing or incorrectly formatted, return zeros (edge case handling)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            
        # If mask happens to be RGB, we typically take the first channel if IDs are just stored across or it needs special decoding.
        # But per standard semantic datasets, it will be 1 channel containing pixel mapping.
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] # Assuming grayscale-like values stored as 3 channels
            
        # Remap non-contiguous values (100, 200, 300, 7100, 10000) to 0-9
        remapped_mask = np.zeros_like(mask, dtype=np.int64)
        for original_id, new_id in self.class_mapping.items():
            remapped_mask[mask == original_id] = new_id
            
        # Determine which transform pipeline to use dynamically
        is_hard_example = img_name in self.hard_mining_set
        active_transforms = self.hard_transforms if (is_hard_example and self.hard_transforms) else self.transforms
            
        if active_transforms:
            augmented = active_transforms(image=image, mask=remapped_mask)
            image = augmented['image']
            remapped_mask = augmented['mask']
            
        return image, remapped_mask.long(), img_name

def get_train_transforms(img_height, img_width):
    # Strong augmentation pipeline per user requirements
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), # In desert off-road, vertical flip might be rare but good for noise resistance
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=img_height, min_width=img_width, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(height=img_height, width=img_width),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_hard_transforms(img_height, img_width):
    # Strong augmentation pipeline exclusively for mined hard examples
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=30, p=0.6), # Intense geometric shifts
        A.PadIfNeeded(min_height=img_height, min_width=img_width, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(height=img_height, width=img_width),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7), # Intense color jitter
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Heavier noise
        A.ElasticTransform(alpha=1.5, sigma=50, p=0.4), # Heavier elastic scaling
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3), # Erase parts randomly
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(img_height, img_width):
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def compute_dataset_statistics(dataset, num_classes, save_path="dataset_stats.json"):
    if os.path.exists(save_path):
        print(f"Loading pre-computed dataset statistics from {save_path}")
        with open(save_path, 'r') as f:
            stats = json.load(f)
        return torch.tensor(stats['class_weights'], dtype=torch.float32), stats['image_weights']

    print("Computing dataset statistics (This may take a minute)...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    image_weights = []

    for i in tqdm(range(len(dataset)), desc="Analyzing masks"):
        _, mask, _ = dataset[i]
        mask_np = mask.numpy()
        
        # Count frequency in this image
        unique, counts = np.unique(mask_np, return_counts=True)
        img_counts = np.zeros(num_classes, dtype=np.int64)
        for val, count in zip(unique, counts):
            if 0 <= val < num_classes:
                img_counts[val] += count
                class_counts[val] += count
                
        # Simple heuristic for image weight: heavily weight images that have less frequent classes.
        # We will compute a more exact weight below, for now just store img_counts
        image_weights.append(img_counts)

    # Compute inverse frequency class weights
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    class_frequencies = class_counts / (class_counts.sum() + epsilon)
    inverse_freq = 1.0 / (class_frequencies + epsilon)
    
    # Normalize per user request: weights / weights.mean()
    class_weights = inverse_freq / inverse_freq.mean()
    
    # Now compute final image weights for sampling
    final_image_weights = []
    for counts in image_weights:
        # Score is sum of (pixels_of_class * weight_of_class)
        # Images with more pixels of rare classes get higher score
        score = np.sum(counts * class_weights)
        final_image_weights.append(float(score))
        
    stats = {
        'class_weights': class_weights.tolist(),
        'image_weights': final_image_weights
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f)
        
    return torch.tensor(class_weights, dtype=torch.float32), final_image_weights
