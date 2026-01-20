# src/utils/prompt_templates.py
"""
Prompt Engineering Templates for LLM Integration
Author: Pranav
Date: January 2025

This module contains carefully crafted prompts for different tasks.
Each prompt is designed to maximize LLM output quality.
"""


class PromptTemplates:
    """
    Collection of prompt templates for vision-language tasks
    """
    
    @staticmethod
    def product_description(caption: str, detected_objects: list = None) -> str:
        """
        Generate e-commerce product description
        
        Design principles:
        1. Clear role definition ("e-commerce specialist")
        2. Structured input (caption + optional detections)
        3. Specific output requirements (length, tone, format)
        4. Examples implicitly shown through instructions
        """
        
        # Build context
        context = f"Image Caption: {caption}"
        
        if detected_objects and len(detected_objects) > 0:
            context += f"\nDetected Objects: {', '.join(detected_objects)}"
        
        prompt = f"""You are an expert e-commerce product description writer.

VISUAL ANALYSIS:
{context}

TASK:
Write a compelling product description (100-150 words) that:

1. **Highlights Key Features**: Based on visual information, emphasize what makes this product distinctive
2. **Uses Benefit-Driven Language**: Focus on how the product serves the customer
3. **Maintains Professional Tone**: Clear, engaging, persuasive without being overly sales-y
4. **Includes Relevant Details**: Color, style, apparent materials, use cases
5. **Optimizes for SEO**: Naturally incorporate product-relevant keywords

REQUIREMENTS:
- Length: 100-150 words
- Format: Single paragraph, no bullet points
- Tone: Professional, engaging, customer-focused
- Accuracy: Only describe what's visible in the analysis

Generate the product description now:"""
        
        return prompt
    
    @staticmethod
    def visual_qa_prompt(caption: str, question: str, detected_objects: list = None) -> str:
        """
        Answer questions about images based on visual analysis
        """
        
        context = f"Caption: {caption}"
        if detected_objects and len(detected_objects) > 0:
            context += f"\nObjects: {', '.join(detected_objects)}"
        
        prompt = f"""Based on this visual analysis, answer the question accurately and concisely.

VISUAL DATA:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the visual evidence provided
- If uncertain, indicate your confidence level
- Be concise but complete
- If the question cannot be answered from the visual data, say so

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def alt_text_generator(caption: str) -> str:
        """
        Generate accessibility-focused alt text
        """
        
        prompt = f"""Generate concise, descriptive alt text for web accessibility.

IMAGE CAPTION: {caption}

REQUIREMENTS:
- Length: 10-15 words maximum
- Describe the essential visual content
- Omit phrases like "image of" or "picture of"
- Focus on what's important for understanding context
- Use clear, simple language

ALT TEXT:"""
        
        return prompt
    
    @staticmethod
    def product_comparison(caption1: str, caption2: str) -> str:
        """
        Compare two product images
        """
        
        prompt = f"""Compare these two product images and highlight key differences.

PRODUCT 1: {caption1}
PRODUCT 2: {caption2}

Provide a structured comparison covering:
1. Visual similarities
2. Key differences
3. Distinct features of each

Keep the comparison concise (50-75 words):"""
        
        return prompt
    
    @staticmethod
    def structured_json_extraction(caption: str, detected_objects: list = None) -> str:
        """
        Extract structured data for downstream processing
        
        This is useful for databases, APIs, search indexing, etc.
        """
        
        context = f"Caption: {caption}"
        if detected_objects:
            context += f"\nDetected: {', '.join(detected_objects)}"
        
        prompt = f"""Extract structured product information in JSON format.

VISUAL DATA:
{context}

Extract the following fields (use null if not determinable):
{{
  "product_type": "primary product category",
  "color": "dominant color or color scheme",
  "style": "style descriptor (modern, vintage, sporty, etc.)",
  "key_features": ["feature1", "feature2", "feature3"],
  "suggested_tags": ["tag1", "tag2", "tag3"],
  "use_case": "primary intended use"
}}

Return ONLY the JSON object, no explanation:"""
        
        return prompt


# Testing function
def test_prompts():
    """Test prompt generation"""
    print("="*70)
    print("🧪 TESTING PROMPT TEMPLATES")
    print("="*70 + "\n")
    
    # Test data
    caption = "a red and white nike running shoe on a white background"
    objects = ["sneaker"]
    
    # Test 1: Product Description
    print("Test 1: Product Description Prompt")
    print("-"*70)
    prompt = PromptTemplates.product_description(caption, objects)
    print(prompt)
    print("\n")
    
    # Test 2: Visual QA
    print("Test 2: Visual QA Prompt")
    print("-"*70)
    question = "What is this product suitable for?"
    prompt = PromptTemplates.visual_qa_prompt(caption, question, objects)
    print(prompt)
    print("\n")
    
    # Test 3: Alt Text
    print("Test 3: Alt Text Generation Prompt")
    print("-"*70)
    prompt = PromptTemplates.alt_text_generator(caption)
    print(prompt)
    print("\n")
    
    print("="*70)
    print("✅ All prompt templates generated successfully!")
    print("="*70)


if __name__ == "__main__":
    test_prompts()