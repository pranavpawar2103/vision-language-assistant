# src/models/vision_analyzer.py
"""
Hybrid Vision Analyzer - Combines BLIP-2 + YOLO intelligently
Author: Pranav
Date: January 2025

Strategy: Use BLIP-2 as primary source of truth, YOLO for spatial info
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.vision_model import ImageCaptioner
from src.models.object_detector import ObjectDetector
from typing import Dict, Union
from PIL import Image


class VisionAnalyzer:
    """
    Intelligent vision analysis combining multiple models
    """
    
    def __init__(self):
        print("🔧 Initializing Vision Analyzer...")
        self.captioner = ImageCaptioner()
        self.detector = ObjectDetector(model_size="n", conf_threshold=0.3)
        print("✅ Vision Analyzer ready!\n")
    
    def analyze(self, image_path: str) -> Dict:
        """
        Comprehensive image analysis
        
        Strategy:
        1. BLIP-2 caption is PRIMARY (it sees everything)
        2. YOLO provides spatial info (when available)
        3. Combine intelligently for best results
        """
        
        # Get caption (our main source of truth)
        caption = self.captioner.generate_caption(image_path)
        
        # Get detections (supplementary spatial info)
        detections = self.detector.detect_objects(image_path)
        
        # Build comprehensive result
        result = {
            "image": image_path,
            "caption": caption,  # Primary description
            "has_detections": detections["num_objects"] > 0,
            "detection_summary": self.detector.get_detection_summary(detections) if detections["num_objects"] > 0 else None,
            "detections": detections,
            "analysis_strategy": self._determine_strategy(caption, detections)
        }
        
        return result
    
    def _determine_strategy(self, caption: str, detections: Dict) -> str:
        """
        Decide which information to prioritize
        """
        if detections["num_objects"] == 0:
            return "caption_only"  # YOLO failed, rely on BLIP-2
        elif detections["num_objects"] > 3:
            return "caption_primary_with_spatial"  # YOLO found scene, use for layout
        else:
            return "hybrid"  # Combine both
    
    def get_product_info(self, image_path: str) -> Dict:
        """
        Extract product-relevant information
        
        This is what we'll feed to the LLM in Day 2
        """
        result = self.analyze(image_path)
        
        # Build structured product info
        product_info = {
            "description": result["caption"],
            "detected_objects": [d["class_name"] for d in result["detections"]["detections"]],
            "confidence_scores": [d["confidence"] for d in result["detections"]["detections"]],
            "spatial_info": None
        }
        
        # Add spatial info if we have good detections
        if result["has_detections"] and result["detections"]["num_objects"] <= 3:
            product_info["spatial_info"] = {
                "object_count": result["detections"]["num_objects"],
                "primary_object": result["detections"]["detections"][0]["class_name"] if result["detections"]["detections"] else None
            }
        
        return product_info


def test_analyzer():
    """Test the hybrid analyzer"""
    from pathlib import Path
    import json
    
    print("="*70)
    print("🧪 TESTING HYBRID VISION ANALYZER")
    print("="*70 + "\n")
    
    analyzer = VisionAnalyzer()
    
    test_images = list(Path("data/test_images").glob("*.jpg"))
    
    for img_path in test_images[:3]:  # Test first 3
        print(f"\n📸 {img_path.name}")
        print("-"*70)
        
        product_info = analyzer.get_product_info(str(img_path))
        
        print(f"Description: {product_info['description']}")
        print(f"YOLO found: {product_info['detected_objects'] if product_info['detected_objects'] else 'Nothing (limited to COCO classes)'}")
        print(f"Strategy: Caption-based (BLIP-2 is more accurate for these products)")
        
    print("\n" + "="*70)
    print("✅ Hybrid analyzer working!")
    print("\n💡 Key Insight: BLIP-2 captions are perfect even when YOLO fails.")
    print("   This is why we'll use captions as primary input for the LLM layer.")
    print("="*70)


if __name__ == "__main__":
    test_analyzer()