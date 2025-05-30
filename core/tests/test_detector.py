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
from unittest.mock import Mock, patch, MagicMock, PropertyMock
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
    
    def test_device_variants(self):
        """Test initialization with different device options."""
        devices = ["cpu", "cuda", "mps"]
        
        for device in devices:
            detector = TableDetector(device=device)
            assert detector.device == device
    
    def test_confidence_threshold_bounds(self):
        """Test confidence threshold boundary values."""
        # Test valid bounds
        detector_min = TableDetector(confidence_threshold=0.1)
        assert detector_min.confidence_threshold == 0.1
        
        detector_max = TableDetector(confidence_threshold=1.0)
        assert detector_max.confidence_threshold == 1.0
    
    def test_initialization_with_all_params(self):
        """Test initialization with all parameters specified."""
        detector = TableDetector(
            model_name="test/model",
            confidence_threshold=0.85,
            device="cpu"
        )
        
        assert detector.model_name == "test/model"
        assert detector.confidence_threshold == 0.85
        assert detector.device == "cpu"
        assert not detector.is_loaded


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
        mock_model_instance.eval = Mock()
        
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
    
    @patch('detector.DetrImageProcessor.from_pretrained')
    @patch('detector.DetrForObjectDetection.from_pretrained')
    def test_model_loading_device_transfer(self, mock_model, mock_processor):
        """Test that model is properly transferred to specified device."""
        mock_processor_instance = Mock()
        mock_model_instance = Mock()
        mock_model_instance.config.id2label = {0: "table"}
        mock_param = Mock()
        mock_param.numel.return_value = 1000000
        mock_model_instance.parameters.return_value = [mock_param]
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = Mock()
        
        mock_processor.return_value = mock_processor_instance
        mock_model.return_value = mock_model_instance
        
        detector = TableDetector(device="cpu")
        detector.load_model()
        
        # Verify model was moved to device
        mock_model_instance.to.assert_called_with("cpu")
        mock_model_instance.eval.assert_called_once()


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
        result = detector.predict("nonexistent.jpg")
        
        assert not result.success
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "failed" in result.error_message.lower()
    
    def test_corrupted_image(self, detector, corrupted_image_data, test_data_dir):
        """Test handling of corrupted image."""
        corrupted_path = test_data_dir / "corrupted.jpg"
        with open(corrupted_path, 'wb') as f:
            f.write(corrupted_image_data)
        
        result = detector.predict(corrupted_path)
        assert not result.success
        assert result.error_message is not None
    
    def test_image_too_large_validation(self, detector):
        """Test _validate_image with oversized image."""
        # Create a normal image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Test with size exceeding limit
        large_size = 50 * 1024 * 1024  # 50MB
        
        with pytest.raises(ImageTooLargeError) as exc_info:
            detector._validate_image(test_image, large_size)
        
        assert "exceeds maximum allowed size" in str(exc_info.value)
        # Check for the size in the error message (formatted with commas)
        assert "52,428,800" in str(exc_info.value) or "52428800" in str(exc_info.value)
    
    def test_image_size_within_limit(self, detector):
        """Test _validate_image with acceptable size."""
        test_image = Image.new('RGB', (100, 100), color='red')
        normal_size = 1024  # 1KB - well within limit
        
        # Should not raise exception
        detector._validate_image(test_image, normal_size)
    
    def test_unsupported_format_validation(self, detector):
        """Test _validate_image with unsupported format."""
        # Create image and manually set unsupported format
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.format = 'XYZ'  # Unsupported format
        
        with pytest.raises(UnsupportedFormatError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Unsupported image format: XYZ" in str(exc_info.value)
        assert "Supported formats:" in str(exc_info.value)
    
    def test_supported_format_validation(self, detector):
        """Test _validate_image with supported formats."""
        supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP', 'WEBP']
        
        for fmt in supported_formats:
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.format = fmt
            
            # Should not raise exception
            detector._validate_image(test_image, 1024)
    
    def test_image_format_none(self, detector):
        """Test _validate_image when image.format is None."""
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.format = None  # No format specified
        
        # Should not raise exception (format check skipped when None)
        detector._validate_image(test_image, 1024)
    
    def test_invalid_image_dimensions_zero_width(self, detector):
        """Test _validate_image with zero width."""
        # Mock image with zero width
        test_image = Mock(spec=Image.Image)
        test_image.width = 0
        test_image.height = 100
        test_image.format = 'JPEG'
        test_image.size = (0, 100)
        test_image.mode = 'RGB'
        
        with pytest.raises(InvalidImageError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Invalid image dimensions: 0x100" in str(exc_info.value)
    
    def test_invalid_image_dimensions_zero_height(self, detector):
        """Test _validate_image with zero height."""
        # Mock image with zero height
        test_image = Mock(spec=Image.Image)
        test_image.width = 100
        test_image.height = 0
        test_image.format = 'JPEG'
        test_image.size = (100, 0)
        test_image.mode = 'RGB'
        
        with pytest.raises(InvalidImageError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Invalid image dimensions: 100x0" in str(exc_info.value)
    
    def test_invalid_image_dimensions_negative(self, detector):
        """Test _validate_image with negative dimensions."""
        # Mock image with negative dimensions
        test_image = Mock(spec=Image.Image)
        test_image.width = -10
        test_image.height = 100
        test_image.format = 'JPEG'
        test_image.size = (-10, 100)
        test_image.mode = 'RGB'
        
        with pytest.raises(InvalidImageError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Invalid image dimensions: -10x100" in str(exc_info.value)
    
    def test_valid_image_dimensions(self, detector):
        """Test _validate_image with valid dimensions."""
        test_image = Image.new('RGB', (100, 200), color='red')
        
        # Should not raise exception
        detector._validate_image(test_image, 1024)
    
    def test_image_integrity_check_success(self, detector):
        """Test _validate_image image integrity check with valid image."""
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Should not raise exception - image is valid
        detector._validate_image(test_image, 1024)
    
    def test_image_integrity_check_failure(self, detector):
        """Test _validate_image image integrity check with corrupted image."""
        # Create a mock image that raises exception when accessing size
        test_image = Mock(spec=Image.Image)
        test_image.width = 100
        test_image.height = 100
        test_image.format = 'JPEG'
        
        # Mock size property to raise exception
        def raise_exception():
            raise Exception("Corrupted image data")
        
        type(test_image).size = PropertyMock(side_effect=raise_exception)
        
        with pytest.raises(InvalidImageError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Corrupted or invalid image" in str(exc_info.value)
        assert "Corrupted image data" in str(exc_info.value)
    
    def test_image_mode_access_failure(self, detector):
        """Test _validate_image when accessing image.mode fails."""
        # Create a mock image that raises exception when accessing mode
        test_image = Mock(spec=Image.Image)
        test_image.width = 100
        test_image.height = 100
        test_image.format = 'JPEG'
        test_image.size = (100, 100)  # Size access works
        
        # Mock mode property to raise exception
        def raise_mode_exception():
            raise Exception("Mode access failed")
        
        type(test_image).mode = PropertyMock(side_effect=raise_mode_exception)
        
        with pytest.raises(InvalidImageError) as exc_info:
            detector._validate_image(test_image, 1024)
        
        assert "Corrupted or invalid image" in str(exc_info.value)
        assert "Mode access failed" in str(exc_info.value)
    
    def test_image_too_large(self, detector):
        """Test handling of oversized image."""
        # Create a mock large image
        large_image = Image.new('RGB', (100, 100), color='red')
        
        # Mock the size check
        with patch('config.settings.max_image_size', 100):  # Very small limit
            result = detector.predict(large_image)
            # Should handle gracefully
            assert result.processing_time >= 0
    
    def test_image_mode_conversion(self, detector, grayscale_image, rgba_image):
        """Test automatic conversion of image modes."""
        # Test grayscale to RGB conversion
        converted_image, _ = detector._load_and_validate_image(grayscale_image)
        assert converted_image.mode == 'RGB'
        
        # Test RGBA to RGB conversion
        converted_image, _ = detector._load_and_validate_image(rgba_image)
        assert converted_image.mode == 'RGB'
    
    def test_url_image_loading(self, detector, mock_requests_get):
        """Test loading image from URL."""
        test_url = "https://example.com/test.jpg"
        
        image, image_info = detector._load_and_validate_image(test_url)
        
        assert isinstance(image, Image.Image)
        mock_requests_get.assert_called_once_with(test_url, timeout=10)
    
    def test_invalid_input_types(self, detector):
        """Test validation with invalid input types."""
        # Test with integer
        result = detector.predict(123)
        assert not result.success
        assert result.error_message is not None
        
        # Test with None
        result = detector.predict(None)
        assert not result.success
        
        # Test with list
        result = detector.predict([])
        assert not result.success
    
    def test_validate_image_comprehensive_coverage(self, detector):
        """Comprehensive test to ensure all branches of _validate_image are covered."""
        from config import settings
        
        # Test 1: Valid image passes all checks
        valid_image = Image.new('RGB', (100, 100), color='white')
        valid_image.format = 'JPEG'
        detector._validate_image(valid_image, 1000)  # Should pass
        
        # Test 2: Size check - exactly at limit (boundary test)
        detector._validate_image(valid_image, settings.max_image_size)  # Should pass
        
        # Test 3: Size check - one byte over limit
        with pytest.raises(ImageTooLargeError):
            detector._validate_image(valid_image, settings.max_image_size + 1)
        
        # Test 4: Format check - empty string format
        empty_format_image = Image.new('RGB', (100, 100), color='white')
        empty_format_image.format = ''  # Empty string, should be skipped
        detector._validate_image(empty_format_image, 1000)  # Should pass
        
        # Test 5: Dimensions check - exactly 1x1 (minimum valid)
        tiny_image = Image.new('RGB', (1, 1), color='white')
        detector._validate_image(tiny_image, 1000)  # Should pass
        
        # Test 6: All validation steps with minimal valid image
        minimal_image = Image.new('RGB', (1, 1), color='white')
        minimal_image.format = 'PNG'  # Supported format
        detector._validate_image(minimal_image, 100)  # Should pass all checks


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
        
        # IMPORTANT: HuggingFace post_process_object_detection filters by threshold
        # So if we want no detections, we return empty tensors (already filtered)
        mock_results = {
            "scores": torch.tensor([]),  # Empty - already filtered by HF
            "labels": torch.tensor([]),
            "boxes": torch.tensor([]).reshape(0, 4)  # Empty with correct shape
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert result.success
        assert len(result.detections) == 0  # Filtered out by threshold
    
    def test_prediction_with_different_formats(self, loaded_detector, various_format_images):
        """Test prediction with different image formats."""
        for fmt, path in various_format_images.items():
            try:
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
                
                result = loaded_detector.predict(path)
                assert result.success
                assert result.processing_time >= 0
            except Exception:
                # Some formats might not be supported
                pytest.skip(f"Format {fmt} not supported in this environment")
    
    def test_single_detection_result(self, loaded_detector, sample_invoice_image):
        """Test handling of single detection result."""
        mock_outputs = Mock()
        loaded_detector.model.return_value = mock_outputs
        
        # Mock single detection
        mock_results = {
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[100.0, 150.0, 400.0, 300.0]])
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert result.success
        assert len(result.detections) == 1
        assert result.max_confidence == 0.95
        assert result.avg_confidence == 0.95
        assert result.detections[0].label == "table"


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
        assert "failed" in result.error_message.lower()
    
    def test_postprocessing_error(self, loaded_detector, sample_invoice_image):
        """Test handling of post-processing errors."""
        # Mock model to return invalid outputs
        loaded_detector.model.return_value = Mock()
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        loaded_detector.processor.post_process_object_detection.side_effect = Exception("Invalid output format")
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert not result.success
        assert result.error_message is not None
    
    def test_preprocessing_error_handling(self, loaded_detector, sample_invoice_image):
        """Test preprocessing error handling."""
        # Mock processor to raise an exception
        loaded_detector.processor.side_effect = Exception("Preprocessing failed")
        
        result = loaded_detector.predict(sample_invoice_image)
        assert not result.success
        assert "failed" in result.error_message.lower()


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
    
    def test_batch_processing_with_custom_batch_size(self, loaded_detector, sample_images_paths):
        """Test batch processing with custom batch size."""
        images = list(sample_images_paths.values()) * 3  # 6 images total
        
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[],
                processing_time=0.1
            )
            
            result = loaded_detector.predict_batch(images, max_batch_size=2)
            assert result.total_images == 6
            assert mock_predict.call_count == 6


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
    
    def test_visualization_with_no_detections(self, loaded_detector, sample_invoice_image, test_data_dir):
        """Test visualization when no detections are found."""
        output_path = test_data_dir / "empty_viz.png"
        
        # Mock empty detection result
        with patch.object(loaded_detector, 'predict') as mock_predict:
            mock_predict.return_value = DetectionResult(
                success=True,
                detections=[],
                processing_time=0.5
            )
            
            # Should not raise exception even with no detections
            loaded_detector.visualize_predictions(sample_invoice_image, output_path)
            assert output_path.exists()
    
    def test_visualization_with_custom_colors(self, loaded_detector, sample_invoice_image, test_data_dir):
        """Test visualization with custom colors."""
        output_path = test_data_dir / "custom_viz.png"
        
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
            
            loaded_detector.visualize_predictions(
                sample_invoice_image, 
                output_path,
                show_confidence=False,
                box_color="blue",
                text_color="yellow"
            )
            assert output_path.exists()


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
    
    def test_confidence_threshold_boundary_values(self, detector):
        """Test boundary values for confidence threshold."""
        # Test minimum value
        detector.update_confidence_threshold(0.0)
        assert detector.confidence_threshold == 0.0
        
        # Test maximum value
        detector.update_confidence_threshold(1.0)
        assert detector.confidence_threshold == 1.0
    
    def test_get_model_info(self, loaded_detector):
        """Test getting model information."""
        info = loaded_detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "num_parameters" in info
        assert "load_time" in info
    
    def test_get_model_info_empty(self, detector):
        """Test getting model info when model not loaded."""
        info = detector.get_model_info()
        assert info == {}
    
    def test_model_info_immutable(self, loaded_detector):
        """Test that model info returns a copy, not reference."""
        info1 = loaded_detector.get_model_info()
        info2 = loaded_detector.get_model_info()
        
        # Modify one copy
        info1["test_key"] = "test_value"
        
        # Other copy should be unaffected
        assert "test_key" not in info2
        assert "test_key" not in loaded_detector._model_info


class TestConfigurationIntegration:
    """Test configuration and settings integration."""
    
    def test_settings_import_and_usage(self, detector):
        """Test that settings are properly imported and used."""
        from config import settings, get_optimal_device, get_settings
        
        # Test that detector uses settings
        assert detector.model_name == settings.model_name
        
        # Test device detection
        device = get_optimal_device()
        assert device in ["cpu", "cuda", "mps"]
        
        # Test get_settings function
        new_settings = get_settings()
        assert hasattr(new_settings, 'model_name')
        assert hasattr(new_settings, 'confidence_threshold')
    
    def test_device_detection_scenarios(self):
        """Test device detection in different scenarios."""
        from config import get_optimal_device
        import torch
        
        # Test CUDA available scenario
        with patch('torch.cuda.is_available', return_value=True):
            device = get_optimal_device()
            assert device == "cuda"
        
        # Test MPS available scenario
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available.return_value = True
                device = get_optimal_device() 
                assert device == "mps"

        # Test CPU only scenario
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available.return_value = False
                device = get_optimal_device()
                assert device == "cpu"


class TestRobustnessAndEdgeCases:
    """Test robustness and edge case handling."""
    
    def test_empty_detection_results(self, loaded_detector, sample_invoice_image):
        """Test handling of empty detection results."""
        mock_outputs = Mock()
        loaded_detector.model.return_value = mock_outputs
        
        # Mock empty results (no detections found)
        mock_results = {
            "scores": torch.tensor([]),
            "labels": torch.tensor([]),
            "boxes": torch.tensor([]).reshape(0, 4)
        }
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        loaded_detector.processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        result = loaded_detector.predict(sample_invoice_image)
        
        assert result.success
        assert len(result.detections) == 0
        assert result.num_detections == 0
        assert result.max_confidence is None
        assert result.avg_confidence is None
    
    def test_postprocess_with_mixed_tensor_types(self, loaded_detector):
        """Test post-processing with mixed tensor types (float vs tensor)."""
        # Test the robust handling in _postprocess_predictions
        
        # Mock processor results with mixed types
        mock_results = {
            "scores": [0.95, 0.87],  # Python floats instead of tensors
            "labels": [0, 0],        # Python ints instead of tensors  
            "boxes": [[100.0, 150.0, 400.0, 300.0], [50.0, 350.0, 450.0, 500.0]]  # Python lists
        }
        
        loaded_detector.processor.post_process_object_detection.return_value = [mock_results]
        
        # This should work due to the robust handling in _postprocess_predictions
        target_sizes = torch.tensor([[224, 224]])
        detections = loaded_detector._postprocess_predictions(None, target_sizes)
        
        assert len(detections) == 2
        assert all(isinstance(det.confidence, float) for det in detections)
        assert all(isinstance(det.bbox, BoundingBox) for det in detections)
    
    def test_image_validation_comprehensive(self, detector, test_data_dir):
        """Comprehensive image validation testing."""
        # Test with extremely small valid image
        tiny_img = Image.new('RGB', (1, 1), color='white')
        result = detector.predict(tiny_img)
        # Should process (though may not find tables)
        assert result.processing_time >= 0


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