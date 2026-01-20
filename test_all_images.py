# create: test_all_images.py
from src.models.object_detector import ObjectDetector
from src.models.vision_model import ImageCaptioner
import os
from pathlib import Path

def quick_test_all():
    """Quick test of all images"""
    
    # Initialize models
    print("🔧 Loading models...\n")
    detector = ObjectDetector(model_size="n", conf_threshold=0.25)
    captioner = ImageCaptioner()
    
    # Get all test images
    test_dir = Path("data/test_images")
    images = list(test_dir.glob("*.jpg"))
    
    print(f"Found {len(images)} test images\n")
    print("="*70)
    
    for img_path in images:
        print(f"\n📸 {img_path.name}")
        print("-"*70)
        
        # Caption
        caption = captioner.generate_caption(str(img_path))
        print(f"Caption: {caption}")
        
        # Detection
        results = detector.detect_objects(str(img_path))
        summary = detector.get_detection_summary(results)
        print(f"Detection: {summary}")
        
        # Draw and save
        output_path = f"assets/{img_path.stem}_annotated.jpg"
        detector.draw_detections(str(img_path), results, save_path=output_path)
        
    print("\n" + "="*70)
    print("✅ All images processed! Check assets/ folder for annotated images.")

if __name__ == "__main__":
    quick_test_all()