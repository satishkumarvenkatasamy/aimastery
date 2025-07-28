#!/usr/bin/env python3
"""
Image Decorator with Plants and Flowers
Analyzes an input image and decorates it with user-specified or random plants/flowers
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import InferenceClient
import random

class ImageDecorator:
    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        self.default_plants = [
            "roses", "sunflowers", "lavender", "daisies", "tulips",
            "ferns", "ivy", "monstera leaves", "cherry blossoms", 
            "hibiscus", "orchids", "jasmine", "lotus flowers"
        ]
    
    def analyze_image(self, image_path):
        """Analyze the image to understand its content and suggest decorations"""
        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe: 1) What's in the image 2) What colors dominate 3) What style of plants/flowers would complement it best 4) Where would be good positions to add decorative elements. Be specific and concise."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"file://{os.path.abspath(image_path)}"
                                }
                            }
                        ]
                    }
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return "Could not analyze image. Using default decoration approach."
    
    def generate_decoration_prompt(self, analysis, user_plants=None):
        """Generate a prompt for creating decorative elements"""
        if user_plants:
            plants = user_plants
        else:
            plants = random.sample(self.default_plants, 3)
        
        plant_list = ", ".join(plants)
        
        prompt = f"""Create beautiful decorative floral elements featuring {plant_list}. 
        Make them artistic, with soft natural colors and organic flowing shapes. 
        The elements should be suitable for overlaying on another image as decorations.
        Style: watercolor, artistic, transparent background effect, delicate and elegant.
        Based on this image analysis: {analysis[:200]}..."""
        
        return prompt, plants
    
    def generate_decorations(self, prompt):
        """Generate decorative plant/flower elements"""
        try:
            decoration = self.client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev"
            )
            return decoration
        except Exception as e:
            print(f"Error generating decorations: {e}")
            return None
    
    def combine_images(self, original_path, decoration_image, output_path):
        """Combine original image with generated decorations"""
        try:
            original = Image.open(original_path)
            
            # Resize decoration to match original proportions
            orig_width, orig_height = original.size
            decoration = decoration_image.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
            
            # Create a composite image
            # Convert to RGBA for transparency support
            if original.mode != 'RGBA':
                original = original.convert('RGBA')
            if decoration.mode != 'RGBA':
                decoration = decoration.convert('RGBA')
            
            # Create multiple decoration layers with different opacities and positions
            result = original.copy()
            
            # Add decoration with reduced opacity
            decoration_alpha = decoration.copy()
            decoration_alpha.putalpha(128)  # 50% transparency
            
            # Blend the images
            result = Image.alpha_composite(result, decoration_alpha)
            
            # Add some subtle decorative borders
            self.add_corner_decorations(result, decoration, orig_width, orig_height)
            
            # Save the result
            result.save(output_path, 'PNG')
            print(f"Decorated image saved as: {output_path}")
            
        except Exception as e:
            print(f"Error combining images: {e}")
    
    def add_corner_decorations(self, result, decoration, width, height):
        """Add small decorative elements to corners"""
        try:
            # Create smaller decoration elements for corners
            corner_size = min(width, height) // 6
            small_decoration = decoration.resize((corner_size, corner_size), Image.Resampling.LANCZOS)
            small_decoration.putalpha(80)  # More transparent
            
            # Add to corners with some offset
            offset = 20
            positions = [
                (offset, offset),  # Top-left
                (width - corner_size - offset, offset),  # Top-right
                (offset, height - corner_size - offset),  # Bottom-left
                (width - corner_size - offset, height - corner_size - offset)  # Bottom-right
            ]
            
            for pos in positions:
                temp = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                temp.paste(small_decoration, pos)
                result.alpha_composite(temp)
                
        except Exception as e:
            print(f"Warning: Could not add corner decorations: {e}")
    
    def decorate_image(self, input_path, output_path=None, user_plants=None):
        """Main method to decorate an image"""
        if not os.path.exists(input_path):
            print(f"Error: Input image '{input_path}' not found.")
            return False
        
        if not output_path:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}_decorated.png"
        
        print(f"Analyzing image: {input_path}")
        analysis = self.analyze_image(input_path)
        print(f"Analysis: {analysis[:150]}...")
        
        print("Generating decoration prompt...")
        decoration_prompt, selected_plants = self.generate_decoration_prompt(analysis, user_plants)
        print(f"Selected plants/flowers: {', '.join(selected_plants)}")
        
        print("Generating decorative elements...")
        decoration = self.generate_decorations(decoration_prompt)
        
        if decoration:
            print("Combining images...")
            self.combine_images(input_path, decoration, output_path)
            return True
        else:
            print("Failed to generate decorations.")
            return False

def main():
    parser = argparse.ArgumentParser(description="Decorate images with plants and flowers")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("-o", "--output", help="Output image path (default: input_decorated.png)")
    parser.add_argument("-p", "--plants", nargs="+", help="Specify plants/flowers to use")
    parser.add_argument("--list-plants", action="store_true", help="List available default plants")
    
    args = parser.parse_args()
    
    decorator = ImageDecorator()
    
    if args.list_plants:
        print("Available default plants/flowers:")
        for plant in decorator.default_plants:
            print(f"  - {plant}")
        return
    
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set.")
        print("Please set your Hugging Face token: export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    success = decorator.decorate_image(
        input_path=args.input_image,
        output_path=args.output,
        user_plants=args.plants
    )
    
    if success:
        print("Image decoration completed successfully!")
    else:
        print("Image decoration failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()