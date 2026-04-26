import cv2
import numpy as np
from pathlib import Path
from typing import List, Generator
import io
import base64


class InputHandler:
    """Handles image and video input processing."""
    
    @staticmethod
    def load_image_from_file(image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in BGR format
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return image
    
    @staticmethod
    def load_image_from_base64(base64_string: str) -> np.ndarray:
        """
        Load image from base64 encoded string.
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            Image as numpy array in BGR format
        """
        # Remove data URI scheme if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array and decode
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode base64 image")
        
        return image
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array in BGR format
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        
        return image
    
    @staticmethod
    def validate_image_file(file_path: str, max_size_mb: int = 50) -> bool:
        """
        Validate image file exists and size is acceptable.
        
        Args:
            file_path: Path to image file
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max {max_size_mb}MB)")
        
        # Check extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format: {path.suffix}")
        
        return True
    
    @staticmethod
    def validate_video_file(file_path: str, max_size_mb: int = 500) -> bool:
        """
        Validate video file exists and size is acceptable.
        
        Args:
            file_path: Path to video file
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max {max_size_mb}MB)")
        
        # Check extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid video format: {path.suffix}")
        
        return True
    
    @staticmethod
    def extract_frames_from_video(
        video_path: str,
        frame_skip: int = 1,
        max_frames: int = None
    ) -> Generator[tuple, None, None]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            frame_skip: Skip every N frames (1 = all frames, 2 = every other frame)
            max_frames: Maximum number of frames to extract (None = all)
            
        Yields:
            Tuple of (frame_index, image_array) for each extracted frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frame_index = 0
        frames_extracted = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_index % frame_skip == 0:
                yield frame_index, frame
                frames_extracted += 1
                
                if max_frames and frames_extracted >= max_frames:
                    break
            
            frame_index += 1
        
        cap.release()
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video file information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        info = {
            'frame_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'frame_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        return info
    
    @staticmethod
    def image_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array in BGR format
            format: Image format ('jpeg' or 'png')
            
        Returns:
            Base64 encoded image string
        """
        if format.lower() == 'jpeg':
            ret, encoded = cv2.imencode('.jpg', image)
        else:
            ret, encoded = cv2.imencode('.png', image)
        
        if not ret:
            raise ValueError("Failed to encode image")
        
        base64_string = base64.b64encode(encoded).decode('utf-8')
        return base64_string
    
    @staticmethod
    def resize_image_to_max_dimension(
        image: np.ndarray,
        max_width: int = 1920,
        max_height: int = 1080
    ) -> np.ndarray:
        """
        Resize image to fit within max dimensions while preserving aspect ratio.
        
        Args:
            image: Image as numpy array
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if w <= max_width and h <= max_height:
            return image
        
        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h))
