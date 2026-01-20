# src/models/vision_model.py
"""
Vision Model Module - BLIP-2 for Image Captioning
Author: Pranav
Date: January 2025

Why BLIP-2?
- State-of-the-art open-source vision-language model
- Efficient: Frozen image encoder + frozen LLM with learnable Q-Former
- Good balance between quality and inference speed
- Can do both captioning and visual question answering
"""

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from typing import Optional, Union
import warnings

warnings.filterwarnings("ignore")


class ImageCaptioner:
    """
    BLIP-2 based image captioner for generating natural language descriptions
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str] = None
    ):
        """
        Initialize the image captioner
        
        Args:
            model_name: HuggingFace model identifier
                Options: 
                - "Salesforce/blip2-opt-2.7b" (faster, 2.7B params)
                - "Salesforce/blip2-flan-t5-xl" (better quality, 3B params)
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
        """
        print("🔧 Initializing BLIP-2 model...")
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"📱 Using device: {self.device}")
        
        # Load processor (handles image preprocessing + tokenization)
        print(f"📥 Loading processor from {model_name}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Load model
        print(f"📥 Loading model from {model_name}...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device
        )
        
        self.model.eval()  # Set to evaluation mode
        
        print("✅ BLIP-2 model loaded successfully!")
        
    def generate_caption(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Generate a natural language caption for an image
        
        Args:
            image: Either a file path (str) or PIL Image
            prompt: Optional text prompt to guide generation
                   e.g., "a photo of" or "Question: What is this? Answer:"
            max_length: Maximum caption length in tokens
            num_beams: Number of beams for beam search (higher = better quality but slower)
        
        Returns:
            Generated caption as string
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Prepare inputs
        if prompt:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        else:
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )
        
        # Decode and clean up
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return caption
    
    def answer_question(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_length: int = 50
    ) -> str:
        """
        Answer a question about an image (Visual Question Answering)
        
        Args:
            image: Either a file path (str) or PIL Image
            question: Question to answer
            max_length: Maximum answer length
        
        Returns:
            Answer as string
        """
        # Format question as prompt
        prompt = f"Question: {question} Answer:"
        
        return self.generate_caption(
            image=image,
            prompt=prompt,
            max_length=max_length
        )
    
    def batch_caption(
        self,
        images: list[Union[str, Image.Image]],
        batch_size: int = 4
    ) -> list[str]:
        """
        Generate captions for multiple images in batches
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images to process at once
        
        Returns:
            List of captions
        """
        captions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process batch
            batch_captions = [
                self.generate_caption(img) for img in batch
            ]
            captions.extend(batch_captions)
        
        return captions


# Testing function
def test_captioner():
    """Test the image captioner with a sample image"""
    import time
    
    print("\n" + "="*60)
    print("🧪 TESTING IMAGE CAPTIONER")
    print("="*60 + "\n")
    
    # Initialize
    captioner = ImageCaptioner()
    
    # Test with sample image
    test_image = "data/test_images/shoe.jpg"
    
    print(f"\n📸 Testing with: {test_image}\n")
    
    # Test 1: Basic caption
    print("Test 1: Basic Caption Generation")
    start = time.time()
    caption = captioner.generate_caption(test_image)
    elapsed = time.time() - start
    print(f"Caption: {caption}")
    print(f"Time: {elapsed:.2f}s")
    
    # Test 2: Guided caption
    print("\nTest 2: Guided Caption (E-commerce style)")
    start = time.time()
    caption = captioner.generate_caption(
        test_image,
        prompt="A product photo of"
    )
    elapsed = time.time() - start
    print(f"Caption: {caption}")
    print(f"Time: {elapsed:.2f}s")
    
    # Test 3: Visual QA
    print("\nTest 3: Visual Question Answering")
    start = time.time()
    answer = captioner.answer_question(
        test_image,
        "What color is this product?"
    )
    elapsed = time.time() - start
    print(f"Q: What color is this product?")
    print(f"A: {answer}")
    print(f"Time: {elapsed:.2f}s")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_captioner()