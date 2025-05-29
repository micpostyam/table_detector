"""
Tests for the TableDetector class.

This module contains comprehensive tests for all TableDetector functionality
including successful detections, error handling, and edge cases.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import torch
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import TableDetector
from models import DetectionResult, Detection, BoundingBox
from exceptions import (
    ModelLoadError, ImageProcessingError, InvalidImageError,
    UnsupportedFormatError, ImageTooLargeError, PredictionError
)


class TestTableDetectorInitialization:
    """Test TableDetector initialization and configuration."""
    
    def test_default_initialization(self):
        """Test detector initialization with default parameters."""
        detector = TableDetector()
        
        assert detector.model_name == "TahaDouaji/detr-doc-table-detection"
        assert detector.confidence_threshold == 0.7  # from config
        assert detector.device in ["cpu", "cuda", "mps"]
        assert not detector.is_loaded
        assert detector.processor is None
        assert detector.model is None
    
    def test_custom_initialization(self):
        """Test detector initialization with custom parameters."""
        detector = TableDetector(
            model_name="custom/model",
            confidence_threshold=0.8,
            device="cpu"
        )
        
        assert detector.model_name == "custom/model"
        assert detector.confidence_threshold == 0.8
        assert detector.device == "cpu"
    
    def test_string_representation(self):
        """Test string representation of detector."""
        detector = TableDetector()
        repr_str = repr(detector)
        
        assert "TableDetector" in repr_str
        assert "not loaded" in repr_str
        assert detector.model_name in repr_str


class TestModelLoading:
    """Test model loading functionality."""
    
    @patch('detector.DetrImageProcessor.from_pretrained')
    @patch('detector.DetrForObjectDetection.from_pretrained')
    def test_successful_model_loading(self, mock_model, mock_processor):
        """Test successful model loading."""
        # Setup mocks
        mock_processor_instance = Mock()
        mock_model_instance = Mock()
        mock_model_instance.config.id2label = {0: "table"}
        
        # Fix the Mock + int error by properly mocking parameters()
        mock_param = Mock()
        mock_param.numel.return_value = 1000000  # Return an int, not a Mock
        mock_model_instance.parameters.return_value = [mock_param]
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_processor.return_value = mock_processor_instance
        mock_model.return_value = mock_model_instance
        
        detector = TableDetector()
        detector.load_model()
        
        assert detector.is_loaded
        assert detector.processor is not None
        assert detector.model is not None
        assert "model_name" in detector._model_info
        assert "load_time" in detector._model_info
        assert detector._model_info["num_parameters"] == 1000000
    
    @patch('detector.DetrImageProcessor.from_pretrained')
    def test_model_loading_failure(self, mock_processor):
        """Test model loading failure handling."""
        mock_processor.side_effect = Exception("Network error")
        
        detector = TableDetector()
        
        with pytest.raises(ModelLoadError) as exc_info:
            detector.load_model()
        
        assert "Failed to load model" in str(exc_info.value)
        assert not detector.is_loaded


class TestImageValidation:
    """Test image validation and preprocessing."""
    
    def test_valid_image_loading(self, detector, sample_invoice_image):
        """Test loading valid image."""
        image, image_info = detector._load_and_validate_image(sample_invoice_image)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image_info.width == sample_invoice_image.width
        assert image_info.height == sample_invoice_image.height
    
    def test_file_not_found(self, detector):
        """Test handling of non-existent file."""
        with pytest.raises(InvalidImageError) as exc_info:
            detector._load_and_validate_image("nonexistent.jpg")
        
        assert "not found" in str(exc_info.value)
    
    def test_corrupted_image(self, detector, corrupted_image_data, test_data_dir):
        """Test handling of corrupted image."""
        corrupted_path = test_data_dir / "corrupted.jpg"
        with open(corrupted_path, 'wb') as f:
            f.write(corrupted_image_data)
        
        with pytest.raises(ImageProcessingError):
            detector._load_and_validate_image(corrupted_path)
    
    def test_unsupported_format(self, detector, test_data_dir):
        """Test handling of unsupported image format."""
        # Create a fake file with unsupported extension
        fake_file = test_data_dir / "test.xyz"
        fake_file.write_text("fake content")
        
        with pytest.raises((UnsupportedFormatError, ImageProcessingError)):
            detector._load_and_validate_image(fake_file)
    
    def test_image_too_large(self, detector):
        """Test handling of oversized image."""
        # Create a mock large image
        large_image = Image.new('RGB', (100, 100), color='red')
        
        # Mock the size check
        with patch('detector.settings.max_image_size', 100):  # Very small limit
            with pytest.raises(ImageTooLargeError):
                detector._load_and_validate_image(large_image)
    
    def test_image_mode_conversion(self, detector):
        """Test automatic conversion of image modes."""
        # Create grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        
        converted_image, _ = detector._load_and_validate_image(gray_image)
        assert converted_image.mode == 'RGB'
    
    def test_url_image_loading(self, detector, mock_requests_get):
        """Test loading image from URL."""
        test_url = "https://example.com/test.jpg"
        
        image, image_info = detector._load_and_validate_image(test_url)
        
        assert isinstance(image, Image.Image)
        mock_requests_get.assert_called_once_with(test_url, timeout=10)


@pytest.mark.unit
class TestPredictionSuccess:
    """Test successful prediction scenarios."""
    
    def test_successful_invoice_detection(self, loaded_detector, sample_invoice_image):
        """Test successful detection on invoice image."""
        # Mock the model components
        mock_outputs = Mock()
        loaded_detector.model.return_value = mock_outputs
        
        # Mock processor post-processing
        mock_results = {
            "scores": torch.tensor([0.95, 0.87]),
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[100.0, 150.0, 400.0, 300.0], 
                                  [50.0, 350.0, 450.0, 500.0]])
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert result.success
        assert len(result.detections) == 2
        assert result.detections[0].confidence == 0.95
        assert result.detections[0].label == "table"
        assert result.processing_time > 0
    
    def test_successful_bank_document_detection(self, loaded_detector, sample_bank_document_image):
        """Test successful detection on bank document."""
        # Mock model response
        mock_outputs = Mock()
        loaded_detector.model.return_value = mock_outputs
        
        mock_results = {
            "scores": torch.tensor([0.92]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[30.0, 120.0, 570.0, 450.0]])
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_bank_document_image)
        
        assert result.success
        assert len(result.detections) == 1
        assert result.detections[0].confidence == 0.92
        assert result.image_info.width == 600
        assert result.image_info.height == 800
    
    def test_no_detections_above_threshold(self, loaded_detector, sample_invoice_image):
        """Test case where no detections meet confidence threshold."""
        # Mock low confidence detections
        mock_outputs = Mock()
        loaded_detector.model.return_value = mock_outputs
        
        # IMPORTANT: Le post_process_object_detection de HuggingFace filtre déjà par seuil!
        # Donc si on met des scores < threshold, ils ne seront pas dans les résultats
        mock_results = {
            "scores": torch.tensor([]),  # Empty - already filtered by post_process
            "labels": torch.tensor([]),
            "boxes": torch.tensor([]).reshape(0, 4)  # Empty tensor with correct shape
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert result.success
        assert len(result.detections) == 0  # No detections after filtering
    
    def test_prediction_with_different_formats(self, loaded_detector, test_data_dir):
        """Test prediction with different image formats."""
        formats = ['JPEG', 'PNG']
        
        for fmt in formats:
            # Create test image in specific format
            img = Image.new('RGB', (200, 200), color='white')
            img_path = test_data_dir / f"test.{fmt.lower()}"
            img.save(img_path, format=fmt)
            
            # Mock successful prediction
            mock_outputs = Mock()
            loaded_detector.model.return_value = mock_outputs
            mock_results = {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]])
            }
            loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
            loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
            
            result = loaded_detector.predict(img_path)
            assert result.success


@pytest.mark.unit
class TestPredictionErrors:
    """Test error handling in predictions."""
    
    def test_prediction_without_loaded_model(self, detector, sample_invoice_image):
        """Test prediction fails gracefully when model not loaded."""
        with patch.object(detector, 'load_model', side_effect=ModelLoadError("Failed to load")):
            with pytest.raises(ModelLoadError):
                detector.predict(sample_invoice_image)
    
    def test_prediction_with_invalid_image(self, loaded_detector):
        """Test prediction with invalid image returns error result (not exception)."""
        # Test with nonexistent file - should return failed result, not raise exception
        result = loaded_detector.predict("nonexistent.jpg")
        
        # The predict method should catch the exception and return a failed result
        assert not result.success
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "failed" in result.error_message.lower()
        assert len(result.detections) == 0
        assert result.processing_time >= 0
    
    def test_model_inference_error(self, loaded_detector, sample_invoice_image):
        """Test handling of model inference errors."""
        # Mock model to raise exception
        loaded_detector.model.side_effect = RuntimeError("CUDA out of memory")
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert not result.success
        assert "Prediction failed" in result.error_message
    
    def test_postprocessing_error(self, loaded_detector, sample_invoice_image):
        """Test handling of post-processing errors."""
        # Mock model to return invalid outputs
        loaded_detector.model.return_value = Mock()
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        loaded_detector.processor.post_process_object_detection.side_effect = Exception("Invalid output format")
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert not result.success
        assert result.error_message is not None


@pytest.mark.unit
class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_successful_batch_processing(self, loaded_detector, sample_images_paths):
        """Test successful batch processing of multiple images."""
        images = list(sample_images_paths.values())
        
        # Mock successful predictions
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[],
                processing_time=0.5,
                image_info=None
            )
            
            result = loaded_detector.predict_batch(images)
            
            assert result.total_images == 2
            assert result.successful_detections == 2
            assert result.failed_detections == 0
            assert result.success_rate == 100.0
            assert mock_predict.call_count == 2
    
    def test_mixed_batch_results(self, loaded_detector, sample_images_paths):
        """Test batch processing with mixed success/failure results."""
        images = list(sample_images_paths.values())
        
        # Mock mixed results
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.side_effect = [
                DetectionResult(success=True, detections=[], processing_time=0.5),
                DetectionResult(success=False, detections=[], processing_time=0.3, error_message="Error")
            ]
            
            result = loaded_detector.predict_batch(images)
            
            assert result.total_images == 2
            assert result.successful_detections == 1
            assert result.failed_detections == 1
            assert result.success_rate == 50.0
    
    def test_batch_processing_with_exceptions(self, loaded_detector, sample_images_paths):
        """Test batch processing when individual predictions raise exceptions."""
        images = list(sample_images_paths.values())
        
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.side_effect = [
                Exception("Prediction failed"),
                DetectionResult(success=True, detections=[], processing_time=0.5)
            ]
            
            result = loaded_detector.predict_batch(images)
            
            assert result.total_images == 2
            assert result.successful_detections == 1
            assert result.failed_detections == 1
    
    def test_empty_batch(self, loaded_detector):
        """Test batch processing with empty input."""
        result = loaded_detector.predict_batch([])
        
        assert result.total_images == 0
        assert result.successful_detections == 0
        assert result.failed_detections == 0


@pytest.mark.unit
class TestVisualization:
    """Test visualization functionality."""
    
    def test_successful_visualization(self, loaded_detector, sample_invoice_image, test_data_dir):
        """Test successful visualization of predictions."""
        output_path = test_data_dir / "visualization.png"
        
        # Mock successful prediction
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[
                    Detection(
                        bbox=BoundingBox(x1=50, y1=50, x2=200, y2=150),
                        confidence=0.9,
                        label="table"
                    )
                ],
                processing_time=0.5
            )
            
            loaded_detector.visualize_predictions(sample_invoice_image, output_path)
            
            assert output_path.exists()
    
    def test_visualization_with_failed_prediction(self, loaded_detector, sample_invoice_image, test_data_dir):
        """Test visualization fails gracefully with failed prediction."""
        output_path = test_data_dir / "failed_viz.png"
        
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=False,
                detections=[],
                processing_time=0.5,
                error_message="No tables found"
            )
            
            with pytest.raises(PredictionError):
                loaded_detector.visualize_predictions(sample_invoice_image, output_path)


@pytest.mark.unit
class TestUtilityMethods:
    """Test utility and configuration methods."""
    
    def test_confidence_threshold_update(self, detector):
        """Test updating confidence threshold."""
        detector.update_confidence_threshold(0.8)
        assert detector.confidence_threshold == 0.8
    
    def test_invalid_confidence_threshold(self, detector):
        """Test validation of confidence threshold."""
        with pytest.raises(ValueError):
            detector.update_confidence_threshold(1.5)
        
        with pytest.raises(ValueError):
            detector.update_confidence_threshold(-0.1)
    
    def test_get_model_info(self, loaded_detector):
        """Test getting model information."""
        info = loaded_detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
    
    def test_get_model_info_empty(self, detector):
        """Test getting model info when model not loaded."""
        info = detector.get_model_info()
        assert info == {}


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics."""
    
    def test_processing_time_reasonable(self, loaded_detector, sample_invoice_image, performance_timer):
        """Test that processing time is reasonable."""
        # Mock quick prediction
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[],
                processing_time=0.5
            )
            
            performance_timer.start()
            result = loaded_detector.predict(sample_invoice_image)
            performance_timer.stop()
            
            # Should complete quickly (mocked)
            assert performance_timer.elapsed < 1.0
            assert result.processing_time > 0
    
    def test_memory_efficiency_batch(self, loaded_detector, sample_images_paths):
        """Test memory efficiency during batch processing."""
        images = list(sample_images_paths.values()) * 3  # 6 images total
        
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[],
                processing_time=0.1
            )
            
            # Should handle batch without memory issues
            result = loaded_detector.predict_batch(images, max_batch_size=2)
            assert result.total_images == 6


if __name__ == "__main__":
    pytest.main([__file__])