"""
Batch processing example for the Table Detection System.

This script demonstrates how to process multiple document images
efficiently using batch processing capabilities.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.detector import TableDetector
from src.models import BatchDetectionResult
from src.exceptions import TableDetectionError


def find_sample_images(directory: str) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Directory path to search for images
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
    image_dir = Path(directory)
    
    if not image_dir.exists():
        print(f"Directory not found: {directory}")
        return []
    
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    return sorted(images)


def process_batch_basic(detector: TableDetector, image_paths: List[Path]) -> BatchDetectionResult:
    """
    Process a batch of images with basic settings.
    
    Args:
        detector: Initialized TableDetector instance
        image_paths: List of image paths to process
        
    Returns:
        BatchDetectionResult containing all results
    """
    print(f"Processing {len(image_paths)} images...")
    
    start_time = time.time()
    
    # Process batch
    batch_result = detector.predict_batch(image_paths, max_batch_size=4)
    
    processing_time = time.time() - start_time
    
    print(f"Batch processing completed in {processing_time:.2f}s")
    print(f"Average time per image: {processing_time/len(image_paths):.2f}s")
    
    return batch_result


def analyze_batch_results(batch_result: BatchDetectionResult) -> Dict[str, Any]:
    """
    Analyze and summarize batch processing results.
    
    Args:
        batch_result: Results from batch processing
        
    Returns:
        Dictionary containing analysis summary
    """
    analysis = {
        "total_images": batch_result.total_images,
        "successful": batch_result.successful_detections,
        "failed": batch_result.failed_detections,
        "success_rate": batch_result.success_rate,
        "total_time": batch_result.total_processing_time,
        "avg_time_per_image": batch_result.avg_processing_time,
        "total_detections": 0,
        "confidence_stats": {
            "min": None,
            "max": None,
            "avg": None
        },
        "detection_counts": {},
        "error_types": {}
    }
    
    all_confidences = []
    
    # Analyze individual results
    for result in batch_result.results:
        if result.success:
            analysis["total_detections"] += len(result.detections)
            
            # Count detections per image
            count = len(result.detections)
            if count not in analysis["detection_counts"]:
                analysis["detection_counts"][count] = 0
            analysis["detection_counts"][count] += 1
            
            # Collect confidence scores
            for detection in result.detections:
                all_confidences.append(detection.confidence)
        else:
            # Categorize errors
            error_type = "Unknown"
            if result.error_message:
                if "not found" in result.error_message.lower():
                    error_type = "File not found"
                elif "format" in result.error_message.lower():
                    error_type = "Format error"
                elif "size" in result.error_message.lower():
                    error_type = "Size error"
                elif "corrupted" in result.error_message.lower():
                    error_type = "Corrupted image"
            
            if error_type not in analysis["error_types"]:
                analysis["error_types"][error_type] = 0
            analysis["error_types"][error_type] += 1
    
    # Calculate confidence statistics
    if all_confidences:
        analysis["confidence_stats"]["min"] = min(all_confidences)
        analysis["confidence_stats"]["max"] = max(all_confidences)
        analysis["confidence_stats"]["avg"] = sum(all_confidences) / len(all_confidences)
    
    return analysis


def save_results_to_json(batch_result: BatchDetectionResult, output_path: str):
    """
    Save batch results to JSON file.
    
    Args:
        batch_result: Results to save
        output_path: Path to save JSON file
    """
    # Convert results to serializable format
    results_data = {
        "batch_summary": {
            "total_images": batch_result.total_images,
            "successful_detections": batch_result.successful_detections,
            "failed_detections": batch_result.failed_detections,
            "success_rate": batch_result.success_rate,
            "total_processing_time": batch_result.total_processing_time,
            "avg_processing_time": batch_result.avg_processing_time
        },
        "individual_results": []
    }
    
    for i, result in enumerate(batch_result.results):
        result_data = {
            "image_index": i,
            "success": result.success,
            "processing_time": result.processing_time,
            "error_message": result.error_message
        }
        
        if result.success:
            result_data["detections"] = [
                {
                    "bbox": detection.bbox.to_list(),
                    "confidence": detection.confidence,
                    "label": detection.label
                }
                for detection in result.detections
            ]
            
            if result.image_info:
                result_data["image_info"] = {
                    "width": result.image_info.width,
                    "height": result.image_info.height,
                    "format": result.image_info.format,
                    "size_bytes": result.image_info.size_bytes,
                    "channels": result.image_info.channels
                }
        
        results_data["individual_results"].append(result_data)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def compare_confidence_thresholds(detector: TableDetector, image_paths: List[Path]):
    """
    Compare results with different confidence thresholds.
    
    Args:
        detector: TableDetector instance
        image_paths: List of images to test
    """
    print("\n=== Confidence Threshold Comparison ===\n")
    
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    # Limit to first few images for this comparison
    test_images = image_paths[:3] if len(image_paths) > 3 else image_paths
    
    comparison_results = []
    
    for threshold in thresholds:
        print(f"Testing threshold: {threshold}")
        detector.update_confidence_threshold(threshold)
        
        batch_result = detector.predict_batch(test_images)
        analysis = analyze_batch_results(batch_result)
        
        comparison_results.append({
            "threshold": threshold,
            "total_detections": analysis["total_detections"],
            "avg_confidence": analysis["confidence_stats"]["avg"],
            "success_rate": analysis["success_rate"]
        })
    
    # Display comparison
    print("\nThreshold Comparison Results:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Detections':<12} {'Avg Conf':<12} {'Success %':<12}")
    print("-" * 60)
    
    for result in comparison_results:
        avg_conf = result["avg_confidence"]
        avg_conf_str = f"{avg_conf:.3f}" if avg_conf else "N/A"
        
        print(f"{result['threshold']:<12} {result['total_detections']:<12} {avg_conf_str:<12} {result['success_rate']:<12.1f}")


def main():
    """Main function demonstrating batch processing."""
    print("Table Detection System - Batch Processing Example")
    print("=" * 55)
    
    # Configuration
    input_directory = "sample_images"  # Replace with your image directory
    output_json = "batch_results.json"
    confidence_threshold = 0.7
    
    print(f"Configuration:")
    print(f"  Input directory: {input_directory}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Output file: {output_json}")
    
    # Find images
    print(f"\n1. Searching for images in '{input_directory}'...")
    image_paths = find_sample_images(input_directory)
    
    if not image_paths:
        print("❌ No images found!")
        print("\n💡 To run this example:")
        print("   1. Create a directory called 'sample_images'")
        print("   2. Add some invoice/document images (JPEG, PNG, etc.)")
        print("   3. Run this script again")
        return
    
    print(f"✅ Found {len(image_paths)} images:")
    for i, path in enumerate(image_paths[:5], 1):  # Show first 5
        print(f"   {i}. {path.name}")
    if len(image_paths) > 5:
        print(f"   ... and {len(image_paths) - 5} more")
    
    # Initialize detector
    print(f"\n2. Initializing TableDetector...")
    detector = TableDetector(confidence_threshold=confidence_threshold)
    
    try:
        detector.load_model()
        print("✅ Model loaded successfully!")
    except TableDetectionError as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Process batch
    print(f"\n3. Processing batch of {len(image_paths)} images...")
    try:
        batch_result = process_batch_basic(detector, image_paths)
        
        # Analyze results
        print(f"\n4. Analyzing results...")
        analysis = analyze_batch_results(batch_result)
        
        # Display summary
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total images processed: {analysis['total_images']}")
        print(f"   Successful detections: {analysis['successful']}")
        print(f"   Failed detections: {analysis['failed']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        print(f"   Total processing time: {analysis['total_time']:.2f}s")
        print(f"   Average time per image: {analysis['avg_time_per_image']:.2f}s")
        
        if analysis['total_detections'] > 0:
            print(f"\n📋 Detection Statistics:")
            print(f"   Total tables detected: {analysis['total_detections']}")
            
            if analysis['confidence_stats']['avg']:
                print(f"   Confidence range: {analysis['confidence_stats']['min']:.3f} - {analysis['confidence_stats']['max']:.3f}")
                print(f"   Average confidence: {analysis['confidence_stats']['avg']:.3f}")
            
            print(f"   Detections per image:")
            for count, num_images in sorted(analysis['detection_counts'].items()):
                print(f"     {count} table(s): {num_images} image(s)")
        
        if analysis['error_types']:
            print(f"\n❌ Error Summary:")
            for error_type, count in analysis['error_types'].items():
                print(f"   {error_type}: {count} image(s)")
        
        # Save results
        print(f"\n5. Saving results...")
        save_results_to_json(batch_result, output_json)
        
        # Optional: Compare thresholds
        if len(image_paths) >= 2:
            compare_confidence_thresholds(detector, image_paths)
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return
    
    print(f"\n✅ Batch processing completed successfully!")
    print(f"\n💡 Next steps:")
    print(f"   - Review results in: {output_json}")
    print(f"   - Adjust confidence threshold if needed")
    print(f"   - Use visualize_predictions() to inspect specific images")


def demo_with_sample_data():
    """
    Create a demo with generated sample data if no real images available.
    """
    print("\n" + "=" * 55)
    print("Demo Mode - Using Generated Sample Data")
    print("=" * 55)
    
    # This would create synthetic test data
    print("Creating synthetic test images...")
    
    from PIL import Image, ImageDraw
    import tempfile
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    # Generate sample images
    sample_images = []
    
    for i in range(3):
        # Create simple document-like image
        img = Image.new('RGB', (600, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some table-like structures
        y_offset = 100 + (i * 50)
        
        # Header
        draw.rectangle([50, y_offset, 550, y_offset + 40], outline='black', width=2)
        
        # Table rows
        for row in range(4):
            y = y_offset + 40 + (row * 30)
            draw.rectangle([50, y, 550, y + 30], outline='black', width=1)
            
            # Columns
            for col in range(1, 4):
                x = 50 + (col * 125)
                draw.line([x, y_offset + 40, x, y + 30], fill='black', width=1)
        
        # Save image
        img_path = temp_dir / f"sample_document_{i+1}.png"
        img.save(img_path)
        sample_images.append(img_path)
    
    print(f"Generated {len(sample_images)} sample images")
    
    # Initialize detector
    detector = TableDetector(confidence_threshold=0.5)
    
    try:
        print("Loading model...")
        detector.load_model()
        
        # Process the sample images
        print("Processing sample images...")
        batch_result = detector.predict_batch(sample_images)
        
        # Show results
        analysis = analyze_batch_results(batch_result)
        print(f"\nDemo Results:")
        print(f"  Processed: {analysis['total_images']} images")
        print(f"  Success rate: {analysis['success_rate']:.1f}%")
        print(f"  Total detections: {analysis['total_detections']}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary files")
        
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Batch processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n🔧 Running demo mode with sample data...")
        demo_with_sample_data()