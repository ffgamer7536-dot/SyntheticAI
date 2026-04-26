import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add Gaia project to path to import model components
gaia_path = Path(__file__).parent.parent.parent / "Gaia-Devcation-2026-main"
sys.path.insert(0, str(gaia_path))

from src.model import create_model
from src.utils import load_config


class InferenceService:
    """
    Handles model loading, preprocessing, and inference for segmentation.
    Singleton pattern - loads model once and reuses.
    """
    
    _instance = None
    _model = None
    _device = None
    _config = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path=None, weights_path=None):
        """
        Initialize inference service.
        
        Args:
            config_path: Path to config.yaml (default: Gaia-Devcation-2026-main/config.yaml)
            weights_path: Path to model weights .pth file (default: Gaia-Devcation-2026-main/models/best.pth)
        """
        if self._model_loaded:
            return
        
        # Set default paths
        if config_path is None:
            config_path = gaia_path / "config.yaml"
        if weights_path is None:
            weights_path = gaia_path / "models" / "best.pth"
        
        self.config_path = str(config_path)
        self.weights_path = str(weights_path)
        
        # Load configuration
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        self._config = load_config(self.config_path)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self._device}")
        print(f"Loading config from: {self.config_path}")
        
        # Load model
        self._load_model()
        self._model_loaded = True
    
    def _load_model(self):
        """Load model from weights file."""
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        
        print(f"Loading model weights from: {self.weights_path}")
        
        self._model = create_model(
            arch=self._config['model']['architecture'],
            backbone=self._config['model']['backbone'],
            weights=None,
            in_channels=self._config['model']['in_channels'],
            num_classes=self._config['model']['num_classes']
        ).to(self._device)
        
        # Load pretrained weights
        state_dict = torch.load(self.weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        
        print("Model loaded successfully")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        Resizes to config dimensions and normalizes.
        
        Args:
            image: OpenCV image (BGR format, uint8)
            
        Returns:
            torch.Tensor of shape (1, 3, H, W) ready for model
        """
        h, w = self._config['dataset']['img_height'], self._config['dataset']['img_width']
        
        # Resize image
        img_resized = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self._device)
    
    def infer(self, image: np.ndarray) -> tuple:
        """
        Run inference on image.
        
        Args:
            image: OpenCV image (BGR format, uint8)
            
        Returns:
            Tuple of:
                - predictions: numpy array of shape (H, W) with class indices (0-9)
                - output_logits: torch.Tensor of raw model outputs (1, 10, H, W)
        """
        img_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self._model(img_tensor)
        
        # Get class predictions
        predictions = torch.argmax(outputs, dim=1)[0].cpu().numpy()
        
        return predictions, outputs
    
    def get_config(self) -> dict:
        """Get loaded configuration."""
        return self._config
    
    def get_device(self) -> torch.device:
        """Get device being used."""
        return self._device
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'architecture': self._config['model']['architecture'],
            'backbone': self._config['model']['backbone'],
            'num_classes': self._config['model']['num_classes'],
            'input_height': self._config['dataset']['img_height'],
            'input_width': self._config['dataset']['img_width'],
            'device': str(self._device),
            'classes': self._config['classes']
        }


# Singleton instance
_inference_service = None

def get_inference_service(config_path=None, weights_path=None) -> InferenceService:
    """Get or create the inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(config_path, weights_path)
    return _inference_service
