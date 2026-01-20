# create file: test_integration.py (in project root)
"""
Integration test: Combine BLIP-2 + YOLOv8
"""

from src.models.vision_model import ImageCaptioner
from src.models.object_detector import ObjectDetector
from PIL import Image
import json


def analyze_image(image_path: str):
    """
    Complete vision analysis pipeline
    """
    print("\n" + "="*70)
    print(f"🔍 ANALYZING: {image_path}")
    print("="*70 + "\n")
    
    # Initialize models
    print("Initializing models...")
    captioner = ImageCaptioner()
    detector = ObjectDetector(model_size="n")
    
    # Step 1: Generate caption
    print("\n📝 Step 1: Generating caption...")
    caption = captioner.generate_caption(image_path)
    print(f"Caption: {caption}")
    
    # Step 2: Detect objects
    print("\n🎯 Step 2: Detecting objects...")
    detections = detector.detect_objects(image_path)
    summary = detector.get_detection_summary(detections)
    print(f"{summary}")
    
    # Step 3: Visual QA
    print("\n❓ Step 3: Visual Question Answering...")
    questions = [
        "What is the main object in this image?",
        "What color is it?",
        "Is this suitable for outdoor use?"
    ]
    
    for q in questions:
        answer = captioner.answer_question(image_path, q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
    
    # Step 4: Combined structured output
    print("\n📊 Step 4: Structured Output (JSON)")
    result = {
        "image": image_path,
        "caption": caption,
        "detection_summary": summary,
        "detections": detections,
        "visual_qa": [
            {"question": q, "answer": captioner.answer_question(image_path, q)}
            for q in questions
        ]
    }
    
    print(json.dumps(result, indent=2))
    
    # Save visualization
    output_path = image_path.replace("test_images", "analyzed").replace(".jpg", "_analyzed.jpg")
    detector.draw_detections(image_path, detections, save_path=output_path)
    
    print(f"\n💾 Visualization saved to: {output_path}")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("data/analyzed", exist_ok=True)
    
    # Test with all images
    test_images = [
        "data/test_images/shoe.jpg",
        "data/test_images/watch.jpg",
        "data/test_images/laptop.jpg"
    ]
    
    results = []
    for img_path in test_images:
        if os.path.exists(img_path):
            result = analyze_image(img_path)
            results.append(result)
    
    print("\n🎉 ALL IMAGES ANALYZED!")
    print(f"Total processed: {len(results)}")