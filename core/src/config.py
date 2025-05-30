"""Configuration module for the table detection system."""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field
import torch


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model configuration
    model_name: str = Field(
        default="TahaDouaji/detr-doc-table-detection",
        description="HuggingFace model name for table detection"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
    
    # Device configuration
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to run inference on"
    )
    
    # Image processing
    max_image_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum allowed image size in bytes"
    )
    
    supported_formats: tuple = Field(
        default=("JPEG", "PNG", "TIFF", "BMP", "WEBP"),
        description="Supported image formats"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Performance
    batch_size: int = Field(
        default=4,
        gt=0,
        description="Batch size for batch processing"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_optimal_device() -> str:
    """Determine the optimal device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"


def get_settings() -> Settings:
    """Get application settings."""
    settings = Settings()
    
    return settings


# Global settings instance
settings = get_settings()