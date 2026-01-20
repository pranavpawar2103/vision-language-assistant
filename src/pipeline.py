# src/pipeline.py
"""
Complete Vision-Language Pipeline
Author: Pranav
Date: January 2025

This is the main integration module that combines:
- Vision models (BLIP-2 + YOLO)
- LLM reasoning (Claude)
- Structured outputs
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vision_analyzer import VisionAnalyzer
from src.models.llm_interface import LLMReasoner
from typing import Dict, Optional
import json
import time


class VisionLanguagePipeline:
    """
    End-to-end pipeline for image → product insights
    """
    
    def __init__(self, use_api: bool = True):
        """
        Initialize complete pipeline
        
        Args:
            use_api: Whether to use Claude API (False for development)
        """
        print("🚀 Initializing Vision-Language Pipeline...")
        print("="*70 + "\n")
        
        # Initialize components
        self.vision = VisionAnalyzer()
        self.llm = LLMReasoner() if use_api else None
        
        if not use_api:
            print("💡 Running in VISION-ONLY mode (no LLM)")
        
        print("\n" + "="*70)
        print("✅ Pipeline ready!\n")
    
    def analyze_product(
        self,
        image_path: str,
        tasks: list = ["description", "alt_text", "structured_data"]
    ) -> Dict:
        """
        Complete product analysis
        
        Args:
            image_path: Path to product image
            tasks: List of analysis tasks to perform
                Options: "description", "qa", "alt_text", "structured_data"
        
        Returns:
            Comprehensive analysis results
        """
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🔍 ANALYZING: {Path(image_path).name}")
        print(f"{'='*70}\n")
        
        # Step 1: Vision analysis
        print("📸 Step 1: Vision Analysis...")
        vision_result = self.vision.get_product_info(image_path)
        
        print(f"   Caption: {vision_result['description']}")
        print(f"   Detected: {vision_result['detected_objects'] or 'N/A (COCO limitation)'}")
        
        # Prepare result object
        result = {
            "image": image_path,
            "vision_analysis": vision_result,
            "llm_outputs": {}
        }
        
        # Step 2: LLM enhancement (if available)
        if self.llm:
            print("\n🤖 Step 2: LLM Enhancement...")
            
            caption = vision_result['description']
            objects = vision_result['detected_objects']
            
            if "description" in tasks:
                print("   → Generating product description...")
                result["llm_outputs"]["product_description"] = \
                    self.llm.generate_product_description(caption, objects)
            
            if "alt_text" in tasks:
                print("   → Generating alt text...")
                result["llm_outputs"]["alt_text"] = \
                    self.llm.generate_alt_text(caption)
            
            if "structured_data" in tasks:
                print("   → Extracting structured data...")
                result["llm_outputs"]["structured_data"] = \
                    self.llm.extract_structured_data(caption, objects)
        
        # Timing
        elapsed = time.time() - start_time
        result["processing_time_seconds"] = round(elapsed, 2)
        
        print(f"\n⏱️  Processing Time: {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return result
    
    def batch_analyze(
        self,
        image_paths: list,
        save_results: bool = True,
        output_dir: str = "data/results"
    ) -> list:
        """
        Analyze multiple images in batch
        """
        
        results = []
        
        print(f"\n{'='*70}")
        print(f"📦 BATCH ANALYSIS: {len(image_paths)} images")
        print(f"{'='*70}\n")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing {Path(img_path).name}...")
            
            result = self.analyze_product(img_path)
            results.append(result)
        
        # Save results if requested
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            output_file = Path(output_dir) / "batch_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ BATCH COMPLETE")
        print(f"{'='*70}")
        print(f"Total images: {len(results)}")
        print(f"Total time: {sum(r['processing_time_seconds'] for r in results):.2f}s")
        print(f"Avg per image: {sum(r['processing_time_seconds'] for r in results) / len(results):.2f}s")
        
        if self.llm:
            stats = self.llm.get_usage_stats()
            print(f"\n📊 API Usage:")
            print(f"   Calls: {stats['api_calls']}")
            print(f"   Estimated cost: ${stats['estimated_cost_usd']}")
        
        return results
    
    def print_result(self, result: Dict):
        """
        Pretty print analysis result
        """
        
        print(f"\n{'='*70}")
        print(f"📄 ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        print(f"\n📸 Image: {result['image']}")
        print(f"\n🔍 Vision Analysis:")
        print(f"   {result['vision_analysis']['description']}")
        
        if "llm_outputs" in result and result["llm_outputs"]:
            print(f"\n🤖 AI-Generated Content:")
            
            if "product_description" in result["llm_outputs"]:
                print(f"\n   📝 Product Description:")
                print(f"   {result['llm_outputs']['product_description']}")
            
            if "alt_text" in result["llm_outputs"]:
                print(f"\n   ♿ Alt Text:")
                print(f"   {result['llm_outputs']['alt_text']}")
            
            if "structured_data" in result["llm_outputs"]:
                print(f"\n   📊 Structured Data:")
                print(f"   {json.dumps(result['llm_outputs']['structured_data'], indent=6)}")
        
        print(f"\n{'='*70}\n")


def demo():
    """Demo the complete pipeline"""
    
    # Initialize pipeline
    pipeline = VisionLanguagePipeline(use_api=True)
    
    # Test with sample images
    test_images = list(Path("data/test_images").glob("*.jpg"))[:3]
    
    if not test_images:
        print("❌ No test images found in data/test_images/")
        return
    
    # Single image demo
    print("\n" + "="*70)
    print("DEMO 1: Single Image Analysis")
    print("="*70)
    
    result = pipeline.analyze_product(str(test_images[0]))
    pipeline.print_result(result)
    
    # Batch demo
    print("\n" + "="*70)
    print("DEMO 2: Batch Analysis")
    print("="*70)
    
    results = pipeline.batch_analyze(
        [str(img) for img in test_images],
        save_results=True
    )
    
    print("\n🎉 Demo complete! Check data/results/batch_results.json")


if __name__ == "__main__":
    demo()