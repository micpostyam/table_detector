# core/src/__init__.py
"""
Table Detection System using DETR (Detection Transformer).

A robust system for detecting tables in document images such as invoices 
and bank statements using pre-trained transformer models.
"""

# Import with fallback for both package and direct import
try:
    from .detector import TableDetector
    from .models import (
        DetectionResult, 
        Detection, 
        BoundingBox, 
        ImageInfo, 
        BatchDetectionResult
    )
    from .exceptions import (
        TableDetectionError,
        ModelLoadError,
        ImageProcessingError,
        InvalidImageError,
        UnsupportedFormatError,
        ImageTooLargeError,
        PredictionError,
        NoDetectionError
    )
    from .config import settings
except ImportError:
    # Fallback for direct imports
    from detector import TableDetector
    from models import (
        DetectionResult, 
        Detection, 
        BoundingBox, 
        ImageInfo, 
        BatchDetectionResult
    )
    from exceptions import (
        TableDetectionError,
        ModelLoadError,
        ImageProcessingError,
        InvalidImageError,
        UnsupportedFormatError,
        ImageTooLargeError,
        PredictionError,
        NoDetectionError
    )
    from config import settings

__version__ = "1.0.0"
__author__ = "Test Technique Dataleon"

__all__ = [
    # Main class
    "TableDetector",
    
    # Models
    "DetectionResult",
    "Detection", 
    "BoundingBox",
    "ImageInfo",
    "BatchDetectionResult",
    
    # Exceptions
    "TableDetectionError",
    "ModelLoadError",
    "ImageProcessingError", 
    "InvalidImageError",
    "UnsupportedFormatError",
    "ImageTooLargeError",
    "PredictionError",
    "NoDetectionError",
    
    # Config
    "settings"
]

# ===================================
# core/tests/__init__.py
"""Tests for the table detection system."""

# ===================================
# Main project __init__.py (at root)
"""
DETR Document Table Detection

Test technique pour Dataleon - Détection de tableaux dans les documents
utilisant un modèle DETR pré-entraîné.
"""