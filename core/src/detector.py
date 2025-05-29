"""
Table detection system using DETR (Detection Transformer) model.

This module provides a robust implementation for detecting tables in document images
such as invoices and bank statements using a pre-trained DETR model.
"""

import logging
import time
import io
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
import requests

# Import with fallback for both package and direct import
try:
    from .models import DetectionResult, Detection, BoundingBox, ImageInfo, BatchDetectionResult
    from .exceptions import (
        ModelLoadError, ImageProcessingError, InvalidImageError, 
        UnsupportedFormatError, ImageTooLargeError, PredictionError
    )
    from .config import settings
except ImportError:
    # Fallback for direct imports (when running tests or scripts directly)
    from models import DetectionResult, Detection, BoundingBox, ImageInfo, BatchDetectionResult
    from exceptions import (
        ModelLoadError, ImageProcessingError, InvalidImageError, 
        UnsupportedFormatError, ImageTooLargeError, PredictionError
    )
    from config import settings

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logger = logging.getLogger(__name__)


class TableDetector:
    """
    Detects tables in document images using DETR (Detection Transformer).
    
    This class provides methods to load a pre-trained DETR model and perform
    table detection on individual images or batches of images.
    
    Attributes:
        model_name: Name of the HuggingFace model to use
        confidence_threshold: Minimum confidence score for detections
        device: Device to run inference on (cpu, cuda, mps)
        processor: DETR image processor
        model: DETR model for object detection
        is_loaded: Whether the model has been loaded successfully
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the TableDetector.
        
        Args:
            model_name: HuggingFace model name. Defaults to config setting.
            confidence_threshold: Minimum confidence for detections. Defaults to config setting.
            device: Device for inference. Defaults to config setting.
        """
        self.model_name = model_name or settings.model_name
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold
        self.device = device or settings.device
        
        # Model components
        self.processor: Optional[DetrImageProcessor] = None
        self.model: Optional[DetrForObjectDetection] = None
        self.is_loaded = False
        
        # Model metadata
        self._model_info: Dict[str, Any] = {}
        
        logger.info(f"Initialized TableDetector with model: {self.model_name}")
        logger.info(f"Device: {self.device}, Confidence threshold: {self.confidence_threshold}")
    
    def load_model(self) -> None:
        """
        Load the DETR model and processor.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            
            # Load processor and model
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            
            # Move model to specified device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Store model information
            self._model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "load_time": time.time() - start_time,
                "id2label": self.model.config.id2label
            }
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully in {self._model_info['load_time']:.2f}s")
            logger.info(f"Model has {self._model_info['num_parameters']:,} parameters")
            
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def predict(self, image_input: Union[str, Path, Image.Image]) -> DetectionResult:
        """
        Predict tables in a single image.
        
        Args:
            image_input: Path to image file, PIL Image, or image URL
            
        Returns:
            DetectionResult containing detected tables and metadata
            
        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Load and validate image
            image, image_info = self._load_and_validate_image(image_input)
            
            # Preprocess image
            inputs = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process predictions
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # [height, width]
            detections = self._postprocess_predictions(outputs, target_sizes)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                success=True,
                detections=detections,
                processing_time=processing_time,
                image_info=image_info,
                model_info=self._model_info
            )
            
        except ModelLoadError:
            # Re-raise model loading errors
            raise
        except Exception as e:
            # Catch ALL other exceptions and return failed result
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            processing_time = time.time() - start_time
            
            return DetectionResult(
                success=False,
                detections=[],
                processing_time=processing_time,
                error_message=error_msg,
                model_info=self._model_info
            )
    
    def predict_batch(
        self, 
        image_inputs: List[Union[str, Path, Image.Image]],
        max_batch_size: Optional[int] = None
    ) -> BatchDetectionResult:
        """
        Predict tables in multiple images.
        
        Args:
            image_inputs: List of image paths, PIL Images, or URLs
            max_batch_size: Maximum batch size for processing
            
        Returns:
            BatchDetectionResult containing all individual results
        """
        batch_size = max_batch_size or settings.batch_size
        start_time = time.time()
        
        results = []
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            
            for image_input in batch:
                try:
                    result = self.predict(image_input)
                    results.append(result)
                    
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process image {image_input}: {str(e)}")
                    failed += 1
                    results.append(DetectionResult(
                        success=False,
                        detections=[],
                        processing_time=0.0,
                        error_message=str(e)
                    ))
        
        total_time = time.time() - start_time
        
        return BatchDetectionResult(
            results=results,
            total_processing_time=total_time,
            successful_detections=successful,
            failed_detections=failed
        )
    
    def visualize_predictions(
        self, 
        image_input: Union[str, Path, Image.Image],
        output_path: Union[str, Path],
        show_confidence: bool = True,
        box_color: str = "red",
        text_color: str = "red"
    ) -> None:
        """
        Visualize detection results on image and save to file.
        
        Args:
            image_input: Input image
            output_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            box_color: Color for bounding boxes
            text_color: Color for text labels
        """
        # Get predictions
        result = self.predict(image_input)
        
        if not result.success:
            raise PredictionError(f"Cannot visualize failed prediction: {result.error_message}")
        
        # Load image
        image, _ = self._load_and_validate_image(image_input)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Draw detections
        for detection in result.detections:
            bbox = detection.bbox
            
            # Draw bounding box
            draw.rectangle(
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                outline=box_color,
                width=2
            )
            
            # Draw label with confidence
            if show_confidence:
                label = f"{detection.label}: {detection.confidence:.2f}"
            else:
                label = detection.label
            
            # Get text size for background
            bbox_text = draw.textbbox((bbox.x1, bbox.y1 - 20), label, font=font)
            
            # Draw text background
            draw.rectangle(bbox_text, fill=box_color)
            
            # Draw text
            draw.text(
                (bbox.x1, bbox.y1 - 20),
                label,
                fill=text_color,
                font=font
            )
        
        # Save image
        image.save(output_path)
        logger.info(f"Visualization saved to {output_path}")
    
    def _load_and_validate_image(self, image_input: Union[str, Path, Image.Image]) -> tuple[Image.Image, ImageInfo]:
        """Load and validate an image from various input types."""
        try:
            # Handle different input types
            if isinstance(image_input, Image.Image):
                image = image_input.copy()
                # Estimate size for PIL Image
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                size_bytes = len(img_bytes.getvalue())
            elif isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                
                # Handle URLs
                if str(image_input).startswith(('http://', 'https://')):
                    response = requests.get(str(image_input), timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    size_bytes = len(response.content)
                else:
                    # Handle local files
                    if not image_path.exists():
                        raise InvalidImageError(f"Image file not found: {image_path}")
                    
                    size_bytes = image_path.stat().st_size
                    image = Image.open(image_path)
            else:
                raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image - IMPORTANT: validate BEFORE using image.width/height
            self._validate_image(image, size_bytes)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create image info AFTER conversion to RGB
            image_info = ImageInfo(
                width=image.width,
                height=image.height,
                format=image.format or 'Unknown',
                size_bytes=size_bytes,
                channels=len(image.getbands())
            )
            
            return image, image_info
            
        except (InvalidImageError, UnsupportedFormatError, ImageTooLargeError):
            raise
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}") from e
    
    def _validate_image(self, image: Image.Image, size_bytes: int) -> None:
        """Validate image properties."""
        # Check file size
        if size_bytes > settings.max_image_size:
            raise ImageTooLargeError(
                f"Image size ({size_bytes:,} bytes) exceeds maximum allowed size "
                f"({settings.max_image_size:,} bytes)"
            )
        
        # Check format
        if image.format and image.format not in settings.supported_formats:
            raise UnsupportedFormatError(
                f"Unsupported image format: {image.format}. "
                f"Supported formats: {', '.join(settings.supported_formats)}"
            )
        
        # Check dimensions (use image.width and image.height, not comparison with image object)
        if image.width <= 0 or image.height <= 0:
            raise InvalidImageError(f"Invalid image dimensions: {image.width}x{image.height}")
        
        # Verify image integrity by trying to load pixel data
        try:
            # Just access the size and mode to trigger any loading errors
            _ = image.size
            _ = image.mode
            # Don't call verify() as it can consume the image stream
        except Exception as e:
            raise InvalidImageError(f"Corrupted or invalid image: {str(e)}") from e
    
    def _preprocess_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Preprocess image for model input."""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        except Exception as e:
            raise ImageProcessingError(f"Image preprocessing failed: {str(e)}") from e
    
    def _postprocess_predictions(
        self, 
        outputs: Any, 
        target_sizes: torch.Tensor
    ) -> List[Detection]:
        """Post-process model outputs to extract detections."""
        try:
            # Use processor's post-processing method
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=self.confidence_threshold
            )[0]
            
            detections = []
            
            # Convert results to Detection objects
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Handle both tensor and float types safely
                if hasattr(score, 'item'):
                    confidence_val = round(score.item(), 3)
                else:
                    confidence_val = round(float(score), 3)
                
                if hasattr(label, 'item'):
                    label_val = label.item()
                else:
                    label_val = int(label)
                
                # Handle box coordinates
                if hasattr(box, 'tolist'):
                    bbox_coords = [round(coord, 2) for coord in box.tolist()]
                else:
                    bbox_coords = [round(float(coord), 2) for coord in box]
                
                detection = Detection(
                    bbox=BoundingBox(
                        x1=bbox_coords[0],
                        y1=bbox_coords[1],
                        x2=bbox_coords[2],
                        y2=bbox_coords[3]
                    ),
                    confidence=confidence_val,
                    label=self.model.config.id2label[label_val]
                )
                
                detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.debug(f"Found {len(detections)} detections above threshold {self.confidence_threshold}")
            
            return detections
            
        except Exception as e:
            raise PredictionError(f"Post-processing failed: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self._model_info.copy() if self._model_info else {}
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Update the confidence threshold for future predictions."""
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        old_threshold = self.confidence_threshold
        self.confidence_threshold = new_threshold
        logger.info(f"Updated confidence threshold from {old_threshold} to {new_threshold}")
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"TableDetector(model={self.model_name}, device={self.device}, status={status})"