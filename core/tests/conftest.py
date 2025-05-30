"""
Pytest configuration and fixtures for table detection tests.
"""

import pytest
import tempfile
import shutil
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw
import io
import requests
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import TableDetector
from models import DetectionResult, Detection, BoundingBox, ImageInfo


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_invoice_image():
    """Create a sample invoice-like image with table-like structure."""
    # Create a simple image that resembles an invoice
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw invoice header
    draw.rectangle([50, 50, 750, 100], outline='black', width=2)
    draw.text((60, 70), "INVOICE #12345", fill='black')
    
    # Draw table-like structure
    # Table header
    draw.rectangle([50, 150, 750, 180], outline='black', width=2, fill='lightgray')
    
    # Table rows
    for i in range(4):
        y = 180 + (i * 30)
        draw.rectangle([50, y, 750, y + 30], outline='black', width=1)
        
        # Vertical lines for columns
        for j in range(1, 4):
            x = 50 + (j * 175)
            draw.line([x, 150, x, y + 30], fill='black', width=1)
    
    return img


@pytest.fixture
def sample_bank_document_image():
    """Create a sample bank document image."""
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Bank header
    draw.rectangle([30, 30, 570, 80], outline='blue', width=3)
    draw.text((40, 50), "BANK STATEMENT", fill='blue')
    
    # Account info table
    draw.rectangle([30, 120, 570, 200], outline='black', width=2)
    draw.line([30, 150, 570, 150], fill='black', width=1)  # Header separator
    
    # Transaction table
    draw.rectangle([30, 250, 570, 450], outline='black', width=2)
    
    # Transaction rows
    for i in range(6):
        y = 280 + (i * 25)
        draw.line([30, y, 570, y], fill='black', width=1)
    
    # Vertical separators
    draw.line([130, 250, 130, 450], fill='black', width=1)
    draw.line([350, 250, 350, 450], fill='black', width=1)
    draw.line([450, 250, 450, 450], fill='black', width=1)
    
    return img


@pytest.fixture
def corrupted_image_data():
    """Create corrupted image data."""
    return b"This is not a valid image file content"


@pytest.fixture
def large_image():
    """Create an oversized image for testing size limits."""
    # Create a very large image (simulated)
    img = Image.new('RGB', (5000, 4000), color='red')
    return img


@pytest.fixture
def empty_image():
    """Create an empty/minimal image."""
    img = Image.new('RGB', (1, 1), color='white')
    return img


@pytest.fixture
def detector():
    """Create a TableDetector instance for testing."""
    return TableDetector(confidence_threshold=0.5)


@pytest.fixture
def loaded_detector(detector):
    """Create a pre-loaded TableDetector instance."""
    # Mock the model loading to avoid downloading during tests
    with patch.object(detector, 'load_model') as mock_load:
        detector.is_loaded = True
        detector._model_info = {
            "model_name": "test-model",
            "device": "cpu",
            "num_parameters": 1000000,
            "load_time": 0.5
        }
        # Mock the necessary components
        detector.processor = Mock()
        detector.model = Mock()
        detector.model.config.id2label = {0: "table"}
        detector.model.to = Mock(return_value=detector.model)
        detector.model.eval = Mock()
        
        # IMPORTANT: Ensure confidence threshold is known
        # The detector fixture creates with default threshold (0.7)
        # But for testing, let's make it explicit
        detector.confidence_threshold = 0.5
        
        yield detector


@pytest.fixture
def mock_successful_prediction():
    """Mock a successful prediction result."""
    return DetectionResult(
        success=True,
        detections=[
            Detection(
                bbox=BoundingBox(x1=100, y1=150, x2=400, y2=300),
                confidence=0.95,
                label="table"
            ),
            Detection(
                bbox=BoundingBox(x1=50, y1=350, x2=450, y2=500),
                confidence=0.87,
                label="table"
            )
        ],
        processing_time=1.23,
        image_info=ImageInfo(
            width=800,
            height=600,
            format="JPEG",
            size_bytes=524288,
            channels=3
        ),
        model_info={"model_name": "test-model"}
    )


@pytest.fixture
def mock_failed_prediction():
    """Mock a failed prediction result."""
    return DetectionResult(
        success=False,
        detections=[],
        processing_time=0.5,
        error_message="Test error message"
    )


@pytest.fixture
def sample_images_paths(test_data_dir, sample_invoice_image, sample_bank_document_image):
    """Create sample image files and return their paths."""
    invoice_path = test_data_dir / "invoice.png"
    bank_path = test_data_dir / "bank_statement.png"
    
    sample_invoice_image.save(invoice_path)
    sample_bank_document_image.save(bank_path)
    
    return {
        "invoice": invoice_path,
        "bank": bank_path
    }


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for URL-based image loading."""
    with patch('requests.get') as mock_get:
        # Create a mock response with image data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        # Create a simple test image as bytes
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        mock_response.content = img_bytes.getvalue()
        
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def grayscale_image():
    """Create a grayscale image for testing mode conversion."""
    img = Image.new('L', (200, 150), color=128)  # Grayscale
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 130], outline=255, width=2)
    return img


@pytest.fixture
def rgba_image():
    """Create an RGBA image for testing mode conversion."""
    img = Image.new('RGBA', (200, 150), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 130], outline=(0, 0, 0, 255), width=2)
    return img


@pytest.fixture
def various_format_images(test_data_dir):
    """Create images in various formats for testing."""
    base_img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(base_img)
    draw.rectangle([10, 10, 90, 90], outline='black', width=2)
    
    formats = {
        'JPEG': 'test.jpg',
        'PNG': 'test.png', 
        'TIFF': 'test.tiff',
        'BMP': 'test.bmp',
        'WEBP': 'test.webp'
    }
    
    paths = {}
    for fmt, filename in formats.items():
        path = test_data_dir / filename
        try:
            if fmt == 'JPEG':
                # Convert to RGB for JPEG
                base_img.save(path, fmt, quality=95)
            else:
                base_img.save(path, fmt)
            paths[fmt] = path
        except Exception:
            # Skip formats not supported by this PIL installation
            pass
    
    return paths


@pytest.fixture
def mock_torch_device():
    """Mock torch device detection for testing."""
    import torch
    original_cuda_available = torch.cuda.is_available
    original_mps_available = getattr(torch.backends, 'mps', Mock()).is_available if hasattr(torch.backends, 'mps') else lambda: False
    
    yield {
        'original_cuda': original_cuda_available,
        'original_mps': original_mps_available
    }


# Pytest marks for categorizing tests
pytest_plugins = ["pytest_mock"]

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer fixture for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def env_vars():
    """Fixture for testing with environment variables."""
    original_env = os.environ.copy()
    yield os.environ
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)