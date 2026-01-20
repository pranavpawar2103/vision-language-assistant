# src/models/llm_interface.py
"""
LLM Interface - Claude API Integration
Author: Pranav
Date: January 2025

This module handles all interactions with Claude API.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from anthropic import Anthropic
import os
from dotenv import load_dotenv
from typing import Optional, Dict
import json
from src.utils.prompt_templates import PromptTemplates

# Load environment variables
load_dotenv()


class LLMReasoner:
    """
    Claude-powered reasoning layer for vision-language tasks
    
    Why separate this into its own class?
    1. Modularity: Easy to swap LLM providers (GPT-4, Gemini, etc.)
    2. Cost tracking: Centralize API calls for monitoring
    3. Prompt management: All LLM interactions in one place
    4. Error handling: Graceful degradation if API fails
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """
        Initialize LLM interface
        
        Args:
            api_key: Anthropic API key (defaults to env var)
            model: Claude model to use
            max_tokens: Maximum response length
            temperature: Creativity level (0.0-1.0)
        """
        
        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            print("⚠️  WARNING: No API key found!")
            print("   Set ANTHROPIC_API_KEY in .env file")
            print("   Operating in FALLBACK mode (template-based responses)")
            self.client = None
        else:
            print("🔧 Initializing Claude API...")
            self.client = Anthropic(api_key=self.api_key)
            print(f"✅ Claude API ready! Model: {model}")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Track API usage for cost monitoring
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def generate_product_description(
        self,
        caption: str,
        detected_objects: Optional[list] = None
    ) -> str:
        """
        Generate e-commerce product description
        
        Args:
            caption: Image caption from BLIP-2
            detected_objects: Optional list from YOLO
        
        Returns:
            Product description string
        """
        
        # Build prompt
        prompt = PromptTemplates.product_description(caption, detected_objects)
        
        # Call LLM or use fallback
        if self.client:
            response = self._call_api(prompt)
            return response
        else:
            return self._fallback_description(caption, detected_objects)
    
    def answer_visual_question(
        self,
        caption: str,
        question: str,
        detected_objects: Optional[list] = None
    ) -> str:
        """
        Answer questions about the image
        """
        
        prompt = PromptTemplates.visual_qa_prompt(caption, question, detected_objects)
        
        if self.client:
            return self._call_api(prompt)
        else:
            return self._fallback_qa(caption, question)
    
    def generate_alt_text(self, caption: str) -> str:
        """
        Generate accessibility alt text
        """
        
        prompt = PromptTemplates.alt_text_generator(caption)
        
        if self.client:
            return self._call_api(prompt, max_tokens=50)
        else:
            # Fallback: Just use caption directly
            return caption
    
    def extract_structured_data(
        self,
        caption: str,
        detected_objects: Optional[list] = None
    ) -> Dict:
        """
        Extract structured JSON data from image
        """
        
        prompt = PromptTemplates.structured_json_extraction(caption, detected_objects)
        
        if self.client:
            response = self._call_api(prompt, max_tokens=300)
            
            try:
                # Parse JSON response
                data = json.loads(response)
                return data
            except json.JSONDecodeError:
                print("⚠️  Failed to parse JSON, returning raw response")
                return {"raw_response": response}
        else:
            return self._fallback_structured_data(caption, detected_objects)
    
    def _call_api(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Make API call to Claude
        
        This is the only method that actually calls the API.
        All other methods route through here for centralized tracking.
        """
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track usage
            self.api_calls += 1
            self.total_input_tokens += message.usage.input_tokens
            self.total_output_tokens += message.usage.output_tokens
            
            # Extract text response
            response_text = message.content[0].text
            
            return response_text
        
        except Exception as e:
            print(f"❌ API Error: {e}")
            print("   Falling back to template-based response")
            return f"[API Error - using fallback]"
    
    def get_usage_stats(self) -> Dict:
        """
        Get API usage statistics for cost tracking
        
        Claude pricing (as of Jan 2025):
        - Input: $3 per million tokens
        - Output: $15 per million tokens
        """
        
        input_cost = (self.total_input_tokens / 1_000_000) * 3
        output_cost = (self.total_output_tokens / 1_000_000) * 15
        total_cost = input_cost + output_cost
        
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(total_cost, 4)
        }
    
    # Fallback methods (work without API key)
    
    def _fallback_description(self, caption: str, objects: Optional[list]) -> str:
        """Template-based description when API unavailable"""
        
        object_str = ', '.join(objects) if objects else 'product'
        
        return f"""This {object_str} features {caption.lower()}. 
        
Designed with attention to detail, this item combines style and functionality. Perfect for those seeking quality and aesthetic appeal. The product showcases distinctive characteristics that make it stand out in its category.

Note: Generated using template (API key not configured). For AI-powered descriptions, add your Claude API key to .env file."""
    
    def _fallback_qa(self, caption: str, question: str) -> str:
        """Simple QA fallback"""
        return f"Based on the image showing {caption}, the answer depends on the specific details visible in the scene."
    
    def _fallback_structured_data(self, caption: str, objects: Optional[list]) -> Dict:
        """Simple structured data extraction"""
        
        # Extract basic info from caption
        words = caption.lower().split()
        
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'gray', 'brown']
        found_color = next((c for c in colors if c in words), None)
        
        return {
            "product_type": objects[0] if objects else "unknown",
            "color": found_color,
            "style": None,
            "key_features": [caption],
            "suggested_tags": objects if objects else [],
            "use_case": None,
            "note": "Template-based extraction (API not configured)"
        }


def test_llm_interface():
    """Test LLM interface"""
    print("="*70)
    print("🧪 TESTING LLM INTERFACE")
    print("="*70 + "\n")
    
    # Initialize
    llm = LLMReasoner()
    
    # Test data
    caption = "a red and white nike running shoe on a white background"
    objects = ["sneaker"]
    
    # Test 1: Product Description
    print("Test 1: Product Description")
    print("-"*70)
    description = llm.generate_product_description(caption, objects)
    print(description)
    print("\n")
    
    # Test 2: Visual QA
    print("Test 2: Visual Question Answering")
    print("-"*70)
    answer = llm.answer_visual_question(
        caption,
        "What type of activities is this product suitable for?",
        objects
    )
    print(f"Q: What type of activities is this product suitable for?")
    print(f"A: {answer}")
    print("\n")
    
    # Test 3: Alt Text
    print("Test 3: Alt Text Generation")
    print("-"*70)
    alt_text = llm.generate_alt_text(caption)
    print(f"Alt Text: {alt_text}")
    print("\n")
    
    # Test 4: Structured Data
    print("Test 4: Structured Data Extraction")
    print("-"*70)
    data = llm.extract_structured_data(caption, objects)
    print(json.dumps(data, indent=2))
    print("\n")
    
    # Show usage stats
    print("="*70)
    print("📊 API Usage Statistics")
    print("="*70)
    stats = llm.get_usage_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ LLM interface test complete!")


if __name__ == "__main__":
    test_llm_interface()