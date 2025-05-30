"""
Tests for Pydantic models.

This module tests the data models used throughout the table detection system.
"""

import pytest
import sys
from pathlib import Path
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import (
    BoundingBox, Detection, ImageInfo, DetectionResult, BatchDetectionResult
)


class TestBoundingBox:
    """Test BoundingBox model."""
    
    def test_valid_bounding_box(self):
        """Test creation of valid bounding box."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 100.0
        assert bbox.y2 == 200.0
    
    def test_bounding_box_properties(self):
        """Test computed properties of bounding box."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        
        assert bbox.width == 100.0
        assert bbox.height == 200.0
        assert bbox.area == 20000.0
        assert bbox.center == (60.0, 120.0)
    
    def test_bounding_box_validation(self):
        """Test validation of bounding box coordinates."""
        # x2 must be greater than x1
        with pytest.raises(ValidationError):
            BoundingBox(x1=100.0, y1=20.0, x2=50.0, y2=200.0)
        
        # y2 must be greater than y1
        with pytest.raises(ValidationError):
            BoundingBox(x1=10.0, y1=200.0, x2=100.0, y2=50.0)
    
    def test_bounding_box_format_conversion(self):
        """Test format conversion methods."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        
        # COCO format: [x, y, width, height]
        coco_format = bbox.to_coco_format()
        assert coco_format == [10.0, 20.0, 100.0, 200.0]
        
        # List format: [x1, y1, x2, y2]
        list_format = bbox.to_list()
        assert list_format == [10.0, 20.0, 110.0, 220.0]


class TestDetection:
    """Test Detection model."""
    
    def test_valid_detection(self):
        """Test creation of valid detection."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(bbox=bbox, confidence=0.95, label="table")
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.95
        assert detection.label == "table"
    
    def test_confidence_validation(self):
        """Test validation of confidence score."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        # Valid confidence values
        Detection(bbox=bbox, confidence=0.0, label="table")
        Detection(bbox=bbox, confidence=1.0, label="table")
        Detection(bbox=bbox, confidence=0.5, label="table")
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            Detection(bbox=bbox, confidence=1.5, label="table")
        
        with pytest.raises(ValidationError):
            Detection(bbox=bbox, confidence=-0.1, label="table")


class TestImageInfo:
    """Test ImageInfo model."""
    
    def test_valid_image_info(self):
        """Test creation of valid image info."""
        info = ImageInfo(
            width=800,
            height=600,
            format="JPEG",
            size_bytes=524288,
            channels=3
        )
        
        assert info.width == 800
        assert info.height == 600
        assert info.format == "JPEG"
        assert info.size_bytes == 524288
        assert info.channels == 3
    
    def test_image_info_properties(self):
        """Test computed properties of image info."""
        info = ImageInfo(
            width=800,
            height=600,
            format="JPEG",
            size_bytes=524288,
            channels=3
        )
        
        assert info.aspect_ratio == 800 / 600
        assert info.megapixels == (800 * 600) / 1_000_000
    
    def test_image_info_validation(self):
        """Test validation of image info fields."""
        # Width must be positive
        with pytest.raises(ValidationError):
            ImageInfo(width=0, height=600, format="JPEG", size_bytes=1000, channels=3)
        
        # Height must be positive
        with pytest.raises(ValidationError):
            ImageInfo(width=800, height=0, format="JPEG", size_bytes=1000, channels=3)
        
        # Size must be non-negative
        with pytest.raises(ValidationError):
            ImageInfo(width=800, height=600, format="JPEG", size_bytes=-1, channels=3)
        
        # Channels must be positive
        with pytest.raises(ValidationError):
            ImageInfo(width=800, height=600, format="JPEG", size_bytes=1000, channels=0)


class TestDetectionResult:
    """Test DetectionResult model."""
    
    def test_successful_detection_result(self):
        """Test creation of successful detection result."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(bbox=bbox, confidence=0.95, label="table")
        image_info = ImageInfo(width=800, height=600, format="JPEG", size_bytes=1000, channels=3)
        
        result = DetectionResult(
            success=True,
            detections=[detection],
            processing_time=1.5,
            image_info=image_info
        )
        
        assert result.success is True
        assert len(result.detections) == 1
        assert result.processing_time == 1.5
        assert result.image_info == image_info
        assert result.error_message is None
    
    def test_failed_detection_result(self):
        """Test creation of failed detection result."""
        result = DetectionResult(
            success=False,
            detections=[],
            processing_time=0.5,
            error_message="Processing failed"
        )
        
        assert result.success is False
        assert len(result.detections) == 0
        assert result.error_message == "Processing failed"
    
    def test_detection_result_properties(self):
        """Test computed properties of detection result."""
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        bbox2 = BoundingBox(x1=50.0, y1=60.0, x2=150.0, y2=250.0)
        
        detections = [
            Detection(bbox=bbox1, confidence=0.95, label="table"),
            Detection(bbox=bbox2, confidence=0.87, label="table")
        ]
        
        result = DetectionResult(
            success=True,
            detections=detections,
            processing_time=1.5
        )
        
        assert result.num_detections == 2
        assert result.max_confidence == 0.95
        assert result.avg_confidence == (0.95 + 0.87) / 2
    
    def test_detection_result_properties_empty(self):
        """Test properties with no detections."""
        result = DetectionResult(
            success=True,
            detections=[],
            processing_time=1.0
        )
        
        assert result.num_detections == 0
        assert result.max_confidence is None
        assert result.avg_confidence is None
    
    def test_filter_by_confidence(self):
        """Test filtering detections by confidence threshold."""
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        bbox2 = BoundingBox(x1=50.0, y1=60.0, x2=150.0, y2=250.0)
        bbox3 = BoundingBox(x1=20.0, y1=30.0, x2=120.0, y2=210.0)
        
        detections = [
            Detection(bbox=bbox1, confidence=0.95, label="table"),
            Detection(bbox=bbox2, confidence=0.65, label="table"),
            Detection(bbox=bbox3, confidence=0.85, label="table")
        ]
        
        result = DetectionResult(
            success=True,
            detections=detections,
            processing_time=1.5
        )
        
        # Filter with threshold 0.8
        filtered_result = result.filter_by_confidence(0.8)
        
        assert len(filtered_result.detections) == 2  # 0.95 and 0.85
        assert filtered_result.detections[0].confidence >= 0.8
        assert filtered_result.detections[1].confidence >= 0.8
        assert filtered_result.success == result.success
        assert filtered_result.processing_time == result.processing_time


class TestBatchDetectionResult:
    """Test BatchDetectionResult model."""
    
    def test_batch_result_creation(self):
        """Test creation of batch detection result."""
        # Create individual results
        result1 = DetectionResult(success=True, detections=[], processing_time=1.0)
        result2 = DetectionResult(success=False, detections=[], processing_time=0.5, error_message="Error")
        result3 = DetectionResult(success=True, detections=[], processing_time=1.2)
        
        batch_result = BatchDetectionResult(
            results=[result1, result2, result3],
            total_processing_time=2.7,
            successful_detections=2,
            failed_detections=1
        )
        
        assert len(batch_result.results) == 3
        assert batch_result.total_processing_time == 2.7
        assert batch_result.successful_detections == 2
        assert batch_result.failed_detections == 1
    
    def test_batch_result_properties(self):
        """Test computed properties of batch result."""
        result1 = DetectionResult(success=True, detections=[], processing_time=1.0)
        result2 = DetectionResult(success=False, detections=[], processing_time=0.5, error_message="Error")
        result3 = DetectionResult(success=True, detections=[], processing_time=1.2)
        
        batch_result = BatchDetectionResult(
            results=[result1, result2, result3],
            total_processing_time=2.7,
            successful_detections=2,
            failed_detections=1
        )
        
        assert batch_result.total_images == 3
        assert batch_result.success_rate == (2 / 3) * 100
        assert batch_result.avg_processing_time == 2.7 / 3
    
    def test_batch_result_empty(self):
        """Test batch result with no results."""
        batch_result = BatchDetectionResult(
            results=[],
            total_processing_time=0.0,
            successful_detections=0,
            failed_detections=0
        )
        
        assert batch_result.total_images == 0
        assert batch_result.success_rate == 0.0
        assert batch_result.avg_processing_time == 0.0
    
    def test_batch_result_validation(self):
        """Test validation of batch result fields."""
        # Total processing time must be non-negative
        with pytest.raises(ValidationError):
            BatchDetectionResult(
                results=[],
                total_processing_time=-1.0,
                successful_detections=0,
                failed_detections=0
            )
        
        # Successful detections must be non-negative
        with pytest.raises(ValidationError):
            BatchDetectionResult(
                results=[],
                total_processing_time=0.0,
                successful_detections=-1,
                failed_detections=0
            )
        
        # Failed detections must be non-negative
        with pytest.raises(ValidationError):
            BatchDetectionResult(
                results=[],
                total_processing_time=0.0,
                successful_detections=0,
                failed_detections=-1
            )


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_complete_workflow_models(self):
        """Test complete workflow using all models together."""
        # Create components
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(bbox=bbox, confidence=0.95, label="table")
        image_info = ImageInfo(width=800, height=600, format="JPEG", size_bytes=1000, channels=3)
        
        # Create detection result
        result = DetectionResult(
            success=True,
            detections=[detection],
            processing_time=1.5,
            image_info=image_info
        )
        
        # Create batch result
        batch_result = BatchDetectionResult(
            results=[result],
            total_processing_time=1.5,
            successful_detections=1,
            failed_detections=0
        )
        
        # Verify everything works together
        # CORRECTION: bbox area = (100-10) * (200-20) = 90 * 180 = 16200
        assert batch_result.results[0].detections[0].bbox.area == 16200.0  # Not 18000
        assert batch_result.results[0].image_info.aspect_ratio == 800 / 600
        assert batch_result.success_rate == 100.0
    
    def test_model_serialization(self):
        """Test that models can be serialized to JSON."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(bbox=bbox, confidence=0.95, label="table")
        
        # Should be able to convert to dict
        detection_dict = detection.model_dump()
        assert detection_dict["confidence"] == 0.95
        assert detection_dict["label"] == "table"
        assert "bbox" in detection_dict
        
        # Should be able to recreate from dict
        recreated = Detection(**detection_dict)
        assert recreated.confidence == detection.confidence
        assert recreated.label == detection.label
        assert recreated.bbox.x1 == detection.bbox.x1


if __name__ == "__main__":
    pytest.main([__file__])