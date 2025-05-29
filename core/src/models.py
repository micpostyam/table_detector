"""Pydantic models for the table detection system."""

from typing import List, Optional, Tuple, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import time


class BoundingBox(BaseModel):
    """Represents a bounding box with coordinates and utility methods."""
    
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @field_validator('x2')
    @classmethod
    def x2_must_be_greater_than_x1(cls, v, info):
        if 'x1' in info.data and v <= info.data['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @field_validator('y2')
    @classmethod
    def y2_must_be_greater_than_y1(cls, v, info):
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2 must be greater than y1')
        return v
    
    @property
    def width(self) -> float:
        """Calculate width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculate height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Calculate area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_coco_format(self) -> List[float]:
        """Convert to COCO format [x, y, width, height]."""
        return [self.x1, self.y1, self.width, self.height]
    
    def to_list(self) -> List[float]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]


class Detection(BaseModel):
    """Represents a single table detection."""
    
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    label: str = Field(..., description="Detection label")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "bbox": {
                    "x1": 100.5,
                    "y1": 150.3,
                    "x2": 400.2,
                    "y2": 300.8
                },
                "confidence": 0.95,
                "label": "table"
            }
        }
    }


class ImageInfo(BaseModel):
    """Information about the processed image."""
    
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")
    size_bytes: int = Field(..., ge=0, description="Image size in bytes")
    channels: int = Field(..., gt=0, description="Number of color channels")
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height
    
    @property
    def megapixels(self) -> float:
        """Calculate image size in megapixels."""
        return (self.width * self.height) / 1_000_000


class DetectionResult(BaseModel):
    """Complete result of a table detection operation."""
    
    success: bool = Field(..., description="Whether detection was successful")
    detections: List[Detection] = Field(default_factory=list, description="List of detected tables")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    image_info: Optional[ImageInfo] = Field(None, description="Information about the processed image")
    error_message: Optional[str] = Field(None, description="Error message if detection failed")
    model_info: Optional[dict] = Field(None, description="Model information used for detection")
    
    @property
    def num_detections(self) -> int:
        """Number of detections found."""
        return len(self.detections)
    
    @property
    def max_confidence(self) -> Optional[float]:
        """Maximum confidence score among detections."""
        if not self.detections:
            return None
        return max(detection.confidence for detection in self.detections)
    
    @property
    def avg_confidence(self) -> Optional[float]:
        """Average confidence score among detections."""
        if not self.detections:
            return None
        return sum(detection.confidence for detection in self.detections) / len(self.detections)
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """Create a new result with detections above the specified confidence threshold."""
        filtered_detections = [
            detection for detection in self.detections 
            if detection.confidence >= min_confidence
        ]
        
        return DetectionResult(
            success=self.success,
            detections=filtered_detections,
            processing_time=self.processing_time,
            image_info=self.image_info,
            error_message=self.error_message,
            model_info=self.model_info
        )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "detections": [
                    {
                        "bbox": {"x1": 100, "y1": 150, "x2": 400, "y2": 300},
                        "confidence": 0.95,
                        "label": "table"
                    }
                ],
                "processing_time": 1.23,
                "image_info": {
                    "width": 800,
                    "height": 600,
                    "format": "JPEG",
                    "size_bytes": 524288,
                    "channels": 3
                },
                "error_message": None
            }
        }
    }


class BatchDetectionResult(BaseModel):
    """Result of batch detection operation."""
    
    results: List[DetectionResult] = Field(..., description="Individual detection results")
    total_processing_time: float = Field(..., ge=0.0, description="Total processing time")
    successful_detections: int = Field(..., ge=0, description="Number of successful detections")
    failed_detections: int = Field(..., ge=0, description="Number of failed detections")
    
    @property
    def total_images(self) -> int:
        """Total number of images processed."""
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.successful_detections / self.total_images) * 100
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per image."""
        if self.total_images == 0:
            return 0.0
        return self.total_processing_time / self.total_images