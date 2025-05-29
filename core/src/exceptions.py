"""Custom exceptions for the table detection system."""


class TableDetectionError(Exception):
    """Base exception for table detection errors."""
    pass


class ModelLoadError(TableDetectionError):
    """Raised when model fails to load."""
    pass


class ImageProcessingError(TableDetectionError):
    """Raised when image processing fails."""
    pass


class InvalidImageError(ImageProcessingError):
    """Raised when image is invalid or corrupted."""
    pass


class UnsupportedFormatError(ImageProcessingError):
    """Raised when image format is not supported."""
    pass


class ImageTooLargeError(ImageProcessingError):
    """Raised when image exceeds maximum size limit."""
    pass


class PredictionError(TableDetectionError):
    """Raised when model prediction fails."""
    pass


class NoDetectionError(TableDetectionError):
    """Raised when no tables are detected (if strict mode)."""
    pass