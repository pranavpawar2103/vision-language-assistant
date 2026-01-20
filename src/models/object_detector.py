# src/models/object_detector.py
"""
Object Detection Module - YOLOv8 for precise object localization
Author: Pranav
Date: January 2025

Why YOLOv8?
- State-of-the-art real-time object detection
- Multiple model sizes (nano to extra-large)
- Easy to use API via Ultralytics
- Good balance of speed vs accuracy for web applications
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Union, Optional
import time


class ObjectDetector:
    """
    YOLOv8-based object detector for identifying and localizing objects
    """
    
    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize object detector
        
        Args:
            model_size: YOLO model size
                "n" (nano): 3.2M params, fastest, 6MB
                "s" (small): 11.2M params, balanced
                "m" (medium): 25.9M params, good accuracy
                "l" (large): 43.7M params, higher accuracy
                "x" (xlarge): 68.2M params, best accuracy
            conf_threshold: Minimum confidence score (0-1)
            iou_threshold: IoU threshold for NMS
        """
        print(f"🔧 Initializing YOLOv8{model_size.upper()} detector...")
        
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        model_path = f"yolov8{model_size}.pt"
        print(f"📥 Loading model: {model_path}")
        
        self.model = YOLO(model_path)
        
        # Get class names (COCO dataset - 80 classes)
        self.class_names = self.model.names
        
        print(f"✅ YOLOv8{model_size.upper()} loaded!")
        print(f"   - Classes: {len(self.class_names)} (COCO dataset)")
        print(f"   - Confidence threshold: {conf_threshold}")
        print(f"   - IoU threshold: {iou_threshold}")
        
    def detect_objects(
        self,
        image: Union[str, np.ndarray, Image.Image],
        verbose: bool = False
    ) -> Dict:
        """
        Detect objects in an image
        
        Args:
            image: Image path, numpy array, or PIL Image
            verbose: Print detection details
        
        Returns:
            Dictionary with:
                - num_objects: Number of detected objects
                - detections: List of detection dicts
                - image_shape: (height, width, channels)
                - processing_time: Time taken in seconds
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        
        for box in results.boxes:
            # Extract box data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            
            # Calculate additional metrics
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(confidence), 3),
                "bbox": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(x2), 1),
                    round(float(y2), 1)
                ],
                "center": [round(float(center_x), 1), round(float(center_y), 1)],
                "size": [round(float(width), 1), round(float(height), 1)],
                "area": round(float(area), 1)
            }
            detections.append(detection)
        
        # Sort by confidence (descending)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        processing_time = time.time() - start_time
        
        result = {
            "num_objects": len(detections),
            "detections": detections,
            "image_shape": results.orig_shape,  # (height, width, channels)
            "processing_time": round(processing_time, 3)
        }
        
        if verbose:
            self._print_detections(result)
        
        return result
    
    def draw_detections(
        self,
        image: Union[str, np.ndarray],
        detections: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Image path or numpy array
            detections: Detection results from detect_objects()
            save_path: Optional path to save annotated image
        
        Returns:
            Annotated image as numpy array
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Define colors for different classes (random but consistent)
        np.random.seed(42)
        colors = {}
        
        for det in detections["detections"]:
            class_name = det["class_name"]
            
            # Assign color if not exists
            if class_name not in colors:
                colors[class_name] = tuple(
                    np.random.randint(0, 255, 3).tolist()
                )
            
            color = colors[class_name]
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name} {det['confidence']:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            cv2.rectangle(
                img,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"💾 Saved annotated image to: {save_path}")
        
        return img
    
    def get_detection_summary(self, detections: Dict) -> str:
        """
        Generate human-readable summary of detections
        
        Args:
            detections: Detection results
        
        Returns:
            Summary string
        """
        if detections["num_objects"] == 0:
            return "No objects detected."
        
        # Count objects by class
        class_counts = {}
        for det in detections["detections"]:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Build summary
        summary_parts = []
        for class_name, count in class_counts.items():
            if count == 1:
                summary_parts.append(f"1 {class_name}")
            else:
                summary_parts.append(f"{count} {class_name}s")
        
        summary = f"Detected {detections['num_objects']} object(s): " + ", ".join(summary_parts)
        
        return summary
    
    def _print_detections(self, result: Dict):
        """Pretty print detection results"""
        print(f"\n{'='*60}")
        print(f"🎯 DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Objects found: {result['num_objects']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"Image size: {result['image_shape']}")
        
        if result['num_objects'] > 0:
            print(f"\nDetailed Results:")
            print(f"{'─'*60}")
            for i, det in enumerate(result['detections'], 1):
                print(f"{i}. {det['class_name']}")
                print(f"   Confidence: {det['confidence']}")
                print(f"   BBox: {det['bbox']}")
                print(f"   Center: {det['center']}")
                print(f"   Size: {det['size'][0]:.0f}x{det['size'][1]:.0f}px")
        
        print(f"{'='*60}\n")


def test_detector():
    """Test object detector"""
    print("\n" + "="*60)
    print("🧪 TESTING OBJECT DETECTOR")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = ObjectDetector(model_size="n")
    
    # Test image
    test_image = "data/test_images/shoe.jpg"
    print(f"📸 Testing with: {test_image}\n")
    
    # Detect objects
    results = detector.detect_objects(test_image, verbose=True)
    
    # Generate summary
    summary = detector.get_detection_summary(results)
    print(f"📝 Summary: {summary}\n")
    
    # Draw detections
    annotated = detector.draw_detections(
        test_image,
        results,
        save_path="assets/test_detection.jpg"
    )
    
    print("✅ Test complete! Check assets/test_detection.jpg")


if __name__ == "__main__":
    test_detector()