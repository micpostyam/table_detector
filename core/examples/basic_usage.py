"""
Basic usage example for the Table Detection System.

This script demonstrates how to use the TableDetector class to detect
tables in document images such as invoices and bank statements.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import TableDetector
from exceptions import TableDetectionError


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def basic_detection_example():
    """Demonstrate basic table detection on a single image."""
    print("=== Basic Table Detection Example ===\n")
    
    # Initialize the detector
    print("1. Initializing TableDetector...")
    detector = TableDetector(confidence_threshold=0.7)
    print(f"   Detector: {detector}")
    
    # Load the model (this will download it if not already cached)
    print("\n2. Loading the model...")
    try:
        detector.load_model()
        print("   Model loaded successfully!")
        print(f"   Model info: {detector.get_model_info()}")
    except TableDetectionError as e:
        print(f"   Error loading model: {e}")
        return
    
    # Example with a sample image (you would replace this with your image path)
    image_path = "path/to/your/invoice.jpg"  # Replace with actual path
    
    print(f"\n3. Detecting tables in image: {image_path}")
    
    try:
        # Perform detection
        result = detector.predict(image_path)
        
        # Display results
        if result.success:
            print(f"   ✅ Detection successful!")
            print(f"   📊 Found {result.num_detections} table(s)")
            print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            
            if result.image_info:
                print(f"   🖼️  Image: {result.image_info.width}x{result.image_info.height} pixels")
                print(f"   📏 Size: {result.image_info.size_bytes:,} bytes")
            
            # Show details for each detection
            for i, detection in enumerate(result.detections, 1):
                print(f"\n   Table {i}:")
                print(f"     - Confidence: {detection.confidence:.3f}")
                print(f"     - Label: {detection.label}")
                print(f"     - Bounding box: {detection.bbox.to_list()}")
                print(f"     - Area: {detection.bbox.area:.1f} pixels²")
        else:
            print(f"   ❌ Detection failed: {result.error_message}")
            
    except FileNotFoundError:
        print(f"   ❌ Image file not found: {image_path}")
        print("   💡 Please provide a valid image path to test the detection")
    except TableDetectionError as e:
        print(f"   ❌ Detection error: {e}")


def confidence_threshold_example():
    """Demonstrate how confidence threshold affects results."""
    print("\n\n=== Confidence Threshold Example ===\n")
    
    detector = TableDetector()
    
    # Example thresholds to test
    thresholds = [0.5, 0.7, 0.9]
    image_path = "path/to/your/document.jpg"  # Replace with actual path
    
    print("Testing different confidence thresholds...")
    
    for threshold in thresholds:
        print(f"\n📊 Testing with threshold: {threshold}")
        detector.update_confidence_threshold(threshold)
        
        try:
            result = detector.predict(image_path)
            if result.success:
                print(f"   Found {result.num_detections} detections")
                if result.detections:
                    confidences = [d.confidence for d in result.detections]
                    print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            else:
                print(f"   Detection failed: {result.error_message}")
        except FileNotFoundError:
            print(f"   ❌ Please provide a valid image path for this example")
            break


def visualization_example():
    """Demonstrate visualization of detection results."""
    print("\n\n=== Visualization Example ===\n")
    
    detector = TableDetector()
    
    try:
        detector.load_model()
        
        input_image = "path/to/your/input.jpg"    # Replace with actual path
        output_image = "detection_result.jpg"      # Output will be saved here
        
        print(f"Creating visualization for: {input_image}")
        print(f"Output will be saved as: {output_image}")
        
        # Create visualization
        detector.visualize_predictions(
            image_input=input_image,
            output_path=output_image,
            show_confidence=True,
            box_color="red",
            text_color="white"
        )
        
        print("✅ Visualization created successfully!")
        print(f"   Check the file: {output_image}")
        
    except FileNotFoundError:
        print("❌ Please provide valid image paths for this example")
    except TableDetectionError as e:
        print(f"❌ Visualization failed: {e}")


def error_handling_example():
    """Demonstrate error handling capabilities."""
    print("\n\n=== Error Handling Example ===\n")
    
    detector = TableDetector()
    
    # Test cases for different types of errors
    test_cases = [
        ("nonexistent.jpg", "File not found"),
        ("https://invalid-url.com/image.jpg", "Invalid URL"),
        # Add more test cases as needed
    ]
    
    print("Testing error handling with various invalid inputs...")
    
    for test_input, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Input: {test_input}")
        
        try:
            result = detector.predict(test_input)
            if result.success:
                print("   ✅ Unexpected success!")
            else:
                print(f"   ✅ Handled gracefully: {result.error_message}")
        except Exception as e:
            print(f"   ✅ Exception caught: {type(e).__name__}: {e}")


def main():
    """Run all examples."""
    setup_logging()
    
    print("Table Detection System - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    basic_detection_example()
    confidence_threshold_example()
    visualization_example()
    error_handling_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\n💡 Tips:")
    print("- Replace example image paths with your actual images")
    print("- Try different confidence thresholds for your use case")
    print("- Check the output visualization to verify results")
    print("- Use batch processing for multiple images (see batch_processing.py)")


if __name__ == "__main__":
    main()