# src/ui/app.py
"""
Gradio UI for Vision-Language Assistant
Author: Pranav
Date: January 2025

Interactive web interface for image analysis
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from PIL import Image
import json
from src.pipeline import VisionLanguagePipeline

# Initialize pipeline
print("🚀 Loading models for Gradio UI...")
pipeline = VisionLanguagePipeline(use_api=True)
print("✅ Gradio UI ready!\n")


def analyze_image_ui(image, tasks):
    """
    Gradio interface function
    
    Args:
        image: PIL Image from Gradio
        tasks: List of selected tasks
    
    Returns:
        Formatted markdown output
    """
    
    if image is None:
        return "❌ Please upload an image first!"
    
    try:
        # Save temp image
        temp_path = "data/temp_gradio.jpg"
        image.save(temp_path)
        
        # Map checkbox selections to task names
        task_list = []
        if "Product Description" in tasks:
            task_list.append("description")
        if "Alt Text" in tasks:
            task_list.append("alt_text")
        if "Structured Data" in tasks:
            task_list.append("structured_data")
        
        if not task_list:
            task_list = ["description"]  # Default
        
        # Run analysis
        result = pipeline.analyze_product(temp_path, tasks=task_list)
        
        # Format output as markdown
        output = f"""
# 🔍 Vision Analysis Results

## 📸 Vision Understanding
**Caption:** {result['vision_analysis']['description']}

**Detected Objects:** {', '.join(result['vision_analysis']['detected_objects']) if result['vision_analysis']['detected_objects'] else 'None (limited to COCO classes)'}

---
"""
        
        # Add LLM outputs if available
        if result.get('llm_outputs'):
            output += "\n## 🤖 AI-Generated Content\n\n"
            
            if 'product_description' in result['llm_outputs']:
                output += f"""
### 📝 Product Description
{result['llm_outputs']['product_description']}

---
"""
            
            if 'alt_text' in result['llm_outputs']:
                output += f"""
### ♿ Accessibility Alt Text
`{result['llm_outputs']['alt_text']}`

---
"""
            
            if 'structured_data' in result['llm_outputs']:
                output += f"""
### 📊 Structured Data (JSON)
```json
{json.dumps(result['llm_outputs']['structured_data'], indent=2)}
```

---
"""
        
        # Add stats
        output += f"""
## ⚡ Performance
- **Processing Time:** {result['processing_time_seconds']}s
"""
        
        if pipeline.llm:
            stats = pipeline.llm.get_usage_stats()
            output += f"""- **API Calls This Session:** {stats['api_calls']}
- **Estimated Cost:** ${stats['estimated_cost_usd']:.4f}
"""
        
        return output
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def answer_question_ui(image, question):
    """Visual QA interface"""
    
    if image is None:
        return "❌ Please upload an image first!"
    
    if not question.strip():
        return "❌ Please enter a question!"
    
    try:
        # Save temp image
        temp_path = "data/temp_gradio_qa.jpg"
        image.save(temp_path)
        
        # NEW: Use BLIP-2's VQA directly for better accuracy
        from src.models.vision_model import ImageCaptioner
        captioner = ImageCaptioner()
        
        # Direct visual QA (BLIP-2 looks at the image to answer)
        direct_answer = captioner.answer_question(temp_path, question)
        
        # Also get LLM's reasoning if available
        vision_result = pipeline.vision.get_product_info(temp_path)
        
        llm_answer = None
        if pipeline.llm:
            llm_answer = pipeline.llm.answer_visual_question(
                vision_result['description'],
                question,
                vision_result['detected_objects']
            )
        
        # Build output showing both approaches
        output = f"""
## 🔍 Visual Analysis
**Caption:** {vision_result['description']}
**Detected Objects:** {', '.join(vision_result['detected_objects']) if vision_result['detected_objects'] else 'N/A'}

---

## ❓ Question
{question}

## 💡 Answer (Direct from Vision Model)
{direct_answer}
"""
        
        if llm_answer:
            output += f"""
---

## 🤖 LLM Reasoning (Based on Caption)
{llm_answer}

---

**Note:** The direct vision model answer looks at the actual image, while the LLM answer reasons from the text caption. For visual attribute questions (color, count, etc.), the direct vision model is more reliable.
"""
        
        return output
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(
    title="Vision-Language Assistant",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🔍 Vision-Language Assistant
    
    **AI-powered product image analysis** combining Computer Vision (BLIP-2 + YOLOv8) with Large Language Models (Claude).
    
    Built by **Pranav** | [GitHub](https://github.com/pranavpawar2103/vision-language-assistant)
    """)
    
    with gr.Tabs():
        # Tab 1: Product Analysis
        with gr.Tab("📦 Product Analysis"):
            gr.Markdown("""
            Upload a product image to generate descriptions, alt text, and structured data.
            """)
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Product Image"
                    )
                    
                    task_checkboxes = gr.CheckboxGroup(
                        choices=[
                            "Product Description",
                            "Alt Text",
                            "Structured Data"
                        ],
                        value=["Product Description", "Alt Text"],
                        label="Select Analysis Tasks"
                    )
                    
                    analyze_btn = gr.Button("🚀 Analyze Image", variant="primary")
                
                with gr.Column():
                    output_md = gr.Markdown(label="Results")
            
            analyze_btn.click(
                fn=analyze_image_ui,
                inputs=[input_image, task_checkboxes],
                outputs=output_md
            )
            
            # Example images
            gr.Examples(
                examples=[
                    ["data/test_images/shoe.jpg"],
                    ["data/test_images/laptop.jpg"],
                    ["data/test_images/headphones.jpg"]
                ],
                inputs=input_image,
                label="Example Images"
            )
        
        # Tab 2: Visual QA
        with gr.Tab("❓ Visual Q&A"):
            gr.Markdown("""
            Ask questions about product images and get AI-powered answers.
            """)
            
            with gr.Row():
                with gr.Column():
                    qa_image = gr.Image(
                        type="pil",
                        label="Upload Image"
                    )
                    
                    qa_question = gr.Textbox(
                        label="Your Question",
                        placeholder="What color is this product?",
                        lines=2
                    )
                    
                    qa_btn = gr.Button("💬 Ask Question", variant="primary")
                
                with gr.Column():
                    qa_output = gr.Markdown(label="Answer")
            
            qa_btn.click(
                fn=answer_question_ui,
                inputs=[qa_image, qa_question],
                outputs=qa_output
            )
            
            # Example questions
            gr.Examples(
                examples=[
                    ["data/test_images/shoe.jpg", "What color is this product?"],
                    ["data/test_images/laptop.jpg", "Is this suitable for gaming?"],
                    ["data/test_images/headphones.jpg", "What are the key features?"]
                ],
                inputs=[qa_image, qa_question],
                label="Example Questions"
            )
    
    # Footer
    gr.Markdown("""
    ---
    **Tech Stack:** BLIP-2 • YOLOv8 • Claude 3.5 • FastAPI • Gradio
    
    **Project Highlights:**
    - Hybrid vision approach (caption-first with object detection)
    - Cost-effective LLM integration ($0.007 per analysis)
    - Production-ready API with comprehensive documentation
    """)


# Launch
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎨 LAUNCHING GRADIO UI")
    print("="*70)
    print("\n💡 The interface will open in your browser automatically")
    print("   If not, go to: http://localhost:7860\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )