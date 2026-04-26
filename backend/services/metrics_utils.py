import numpy as np
import torch
import cv2
from typing import Dict, Tuple, List


# Color palette for 10 classes (distinct colors for visualizations)
CLASS_COLORS = {
    0: (0, 255, 0),      # Trees - Green
    1: (34, 139, 34),    # Lush Bushes - Dark Green
    2: (255, 255, 0),    # Dry Grass - Yellow
    3: (210, 180, 140),  # Dry Bushes - Tan
    4: (165, 42, 42),    # Ground Clutter - Brown
    5: (255, 20, 147),   # Flowers - Deep Pink
    6: (160, 82, 45),    # Logs - Sienna
    7: (128, 128, 128),  # Rocks - Gray
    8: (0, 128, 255),    # Landscape - Orange
    9: (0, 0, 255),      # Sky - Blue
}


def compute_per_class_coverage(predictions: np.ndarray) -> Dict[int, float]:
    """
    Compute percentage coverage of each class in prediction map.
    
    Args:
        predictions: numpy array of shape (H, W) with class indices (0-9)
        
    Returns:
        Dictionary mapping class_id -> coverage_percentage (0-100)
    """
    total_pixels = predictions.size
    coverage = {}
    
    for class_id in range(10):  # 10 classes
        num_pixels = np.sum(predictions == class_id)
        coverage[class_id] = (num_pixels / total_pixels) * 100.0
    
    return coverage


def compute_iou_per_class(
    predictions: np.ndarray,
    ground_truth: np.ndarray = None
) -> Tuple[Dict[int, float], float]:
    """
    Compute per-class IoU (Intersection over Union).
    If ground_truth not provided, returns NaN for all classes.
    
    Args:
        predictions: numpy array of shape (H, W) with predicted class indices
        ground_truth: numpy array of shape (H, W) with ground truth class indices, or None
        
    Returns:
        Tuple of:
            - Dictionary mapping class_id -> IoU (0-1 or NaN)
            - mean_iou: Average IoU across classes (NaN if no ground truth)
    """
    class_ious = {}
    
    if ground_truth is None:
        # No ground truth available
        for class_id in range(10):
            class_ious[class_id] = np.nan
        return class_ious, np.nan
    
    # Compute IoU per class
    for class_id in range(10):
        pred_mask = predictions == class_id
        gt_mask = ground_truth == class_id
        
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        
        if union == 0:
            iou = np.nan  # No pixels of this class in image
        else:
            iou = intersection / union
        
        class_ious[class_id] = iou
    
    # Compute mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in class_ious.values() if not np.isnan(iou)]
    mean_iou = np.nanmean(valid_ious) if valid_ious else np.nan
    
    return class_ious, mean_iou


def create_segmentation_overlay(
    original_image: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create a segmentation overlay by blending original image with colored predictions.
    
    Args:
        original_image: numpy array of shape (H, W, 3) in BGR format (uint8)
        predictions: numpy array of shape (H, W) with class indices
        alpha: blending factor (0-1), higher = more segmentation visible
        
    Returns:
        Blended image of shape (H, W, 3) in BGR format (uint8)
    """
    # Create colored segmentation map
    h, w = predictions.shape
    segmentation_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(10):
        mask = predictions == class_id
        color_bgr = CLASS_COLORS[class_id]  # RGB format from dict
        # Convert RGB to BGR for OpenCV
        color_bgr_cv = (color_bgr[2], color_bgr[1], color_bgr[0])
        segmentation_colored[mask] = color_bgr_cv
    
    # Blend original and segmentation
    overlay = cv2.addWeighted(original_image, 1 - alpha, segmentation_colored, alpha, 0)
    
    return overlay


def create_side_by_side_comparison(
    original_image: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """
    Create a side-by-side comparison of original image and segmentation.
    
    Args:
        original_image: numpy array of shape (H, W, 3) in BGR format (uint8)
        predictions: numpy array of shape (H, W) with class indices
        
    Returns:
        Concatenated image of shape (H, 2*W, 3) in BGR format (uint8)
    """
    # Create colored segmentation map
    h, w = predictions.shape
    segmentation_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(10):
        mask = predictions == class_id
        color_bgr = CLASS_COLORS[class_id]
        color_bgr_cv = (color_bgr[2], color_bgr[1], color_bgr[0])
        segmentation_colored[mask] = color_bgr_cv
    
    # Resize original to match segmentation dimensions if needed
    if original_image.shape[:2] != (h, w):
        original_resized = cv2.resize(original_image, (w, h))
    else:
        original_resized = original_image
    
    # Concatenate horizontally
    comparison = np.hstack([original_resized, segmentation_colored])
    
    return comparison


def format_results_for_json(
    predictions: np.ndarray,
    coverage: Dict[int, float],
    class_ious: Dict[int, float],
    mean_iou: float,
    class_names: List[str],
    inference_time_ms: float
) -> dict:
    """
    Format inference results as JSON-serializable dictionary.
    
    Args:
        predictions: numpy array of shape (H, W) with class indices
        coverage: Dict of class_id -> coverage_percentage
        class_ious: Dict of class_id -> IoU value
        mean_iou: Mean IoU value
        class_names: List of class names
        inference_time_ms: Inference time in milliseconds
        
    Returns:
        Dictionary with all results in JSON-serializable format
    """
    results = {
        'inference_time_ms': float(inference_time_ms),
        'metrics': {
            'mean_iou': float(mean_iou) if not np.isnan(mean_iou) else None,
            'per_class_coverage': {},
            'per_class_iou': {}
        },
        'prediction_shape': list(predictions.shape)
    }
    
    # Per-class metrics
    for class_id in range(10):
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class {class_id}'
        
        # Coverage
        results['metrics']['per_class_coverage'][class_name] = float(coverage[class_id])
        
        # IoU
        iou_val = class_ious[class_id]
        results['metrics']['per_class_iou'][class_name] = float(iou_val) if not np.isnan(iou_val) else None
    
    return results


def predictions_to_image(predictions: np.ndarray) -> np.ndarray:
    """
    Convert prediction array to a viewable image (colored by class).
    
    Args:
        predictions: numpy array of shape (H, W) with class indices (0-9)
        
    Returns:
        Image of shape (H, W, 3) in BGR format (uint8)
    """
    h, w = predictions.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(10):
        mask = predictions == class_id
        color_bgr = CLASS_COLORS[class_id]
        color_bgr_cv = (color_bgr[2], color_bgr[1], color_bgr[0])
        image[mask] = color_bgr_cv
    
    return image
