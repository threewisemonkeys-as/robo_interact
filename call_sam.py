#!/usr/bin/env python3

import requests
import base64
import json
from PIL import Image
import io
import numpy as np

# Your ngrok URL
SERVER_URL = "https://9970-128-84-100-13.ngrok-free.app"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def decode_base64_to_image(base64_string):
    """Convert base64 string back to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def generate_mask(image_path, SERVER_URL=SERVER_URL):
    """
    Generate automatic segmentation masks for entire image
    
    Args:
        image_path: Path to image file
    """
    # Encode image to base64
    image_b64 = encode_image_to_base64(image_path)

    data = {
        "image": image_b64,
    }
    
    # Make API call
    response = requests.post(f"{SERVER_URL}/generate_masks", json=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract mask information
        masks_data = []
        for i, mask_info in enumerate(result['masks']):
            # Get the base64 mask from the 'mask' key
            mask_b64 = mask_info['mask']
            # Decode the mask
            mask_image = decode_base64_to_image(mask_b64)
            
            # Store mask with its metadata
            mask_data = {
                'mask_image': mask_image,
                'area': mask_info['area'],
                'bbox': mask_info['bbox'],  # [x, y, width, height]
                'predicted_iou': mask_info['predicted_iou'],
                'stability_score': mask_info['stability_score'],
                'crop_box': mask_info['crop_box']
            }
            masks_data.append(mask_data)
        
        # Return both the masks and metadata
        return {
            'masks': masks_data,
            'num_masks': result['num_masks'],
            'image_scale': result['image_scale'],
            'message': result['message']
        }
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def check_server_status():
    """Check if the server is running and SAM is loaded"""
    response = requests.get(f"{SERVER_URL}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"Server status: {status}")
        return status
    else:
        print(f"Error checking status: {response.status_code}")
        return None

def visualize_masks(original_image_path, masks_data, output_prefix="mask"):
    """
    Save individual masks and create a combined visualization
    
    Args:
        original_image_path: Path to original image
        masks_data: List of mask dictionaries from generate_mask
        output_prefix: Prefix for output files
    """
    # Load original image
    original_image = Image.open(original_image_path)
    
    print(f"Found {len(masks_data)} masks to visualize")
    
    # Save each mask individually
    for i, mask_data in enumerate(masks_data):
        mask_image = mask_data['mask_image']
        
        # Print mask information
        print(f"Mask {i}:")
        print(f"  - Area: {mask_data['area']} pixels")
        print(f"  - Bounding box: {mask_data['bbox']}")
        print(f"  - IoU score: {mask_data['predicted_iou']:.3f}")
        print(f"  - Stability: {mask_data['stability_score']:.3f}")
    
    # Create a combined visualization
    if len(masks_data) > 0:
        # Start with the original image
        combined = original_image.copy().convert('RGBA')
        
        # Overlay all masks with different colors
        colors = [
            (255, 0, 0, 128),    # Red
            (0, 255, 0, 128),    # Green
            (0, 0, 255, 128),    # Blue
            (255, 255, 0, 128),  # Yellow
            (255, 0, 255, 128),  # Magenta
            (0, 255, 255, 128),  # Cyan
        ]
        
        for i, mask_data in enumerate(masks_data):
            mask_image = mask_data['mask_image']
            color = colors[i % len(colors)]
            
            # Create colored overlay
            overlay = Image.new('RGBA', combined.size, (0, 0, 0, 0))
            mask_np = np.array(mask_image)
            
            # Apply color where mask is white
            if len(mask_np.shape) == 2:  # Grayscale mask
                colored_mask = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
                colored_mask[mask_np > 128] = color
            else:  # RGB mask
                # Convert to grayscale for masking
                mask_gray = np.mean(mask_np, axis=2)
                colored_mask = np.zeros((*mask_gray.shape, 4), dtype=np.uint8)
                colored_mask[mask_gray > 128] = color
            
            overlay.paste(Image.fromarray(colored_mask), (0, 0))
            combined = Image.alpha_composite(combined, overlay)
        
        # Save combined visualization
        combined_filename = f"{output_prefix}_combined.png"
        combined.save(combined_filename)
        print(f"\nCombined visualization saved as: {combined_filename}")
        
        return combined_filename
    
    return None

# Example usage
if __name__ == "__main__":
    # Check server status first
    print("Checking server status...")
    status = check_server_status()
    
    if not status or status['status'] != 'ok':
        print("Server is not ready!")
        exit(1)
    
    print("\nExample 1: Generating automatic masks...")
    result = generate_mask(
        image_path="/Users/daiyijia/robo_interact/data/images/color_0_1747009769.png",
    )
    
    if result:
        print(f"\n{result['message']}")
        print(f"Number of masks: {result['num_masks']}")
        print(f"Image scale factor: {result['image_scale']}")
        
        # Visualize all masks
        combined_file = visualize_masks(
            "/Users/daiyijia/robo_interact/visualization.png",
            result['masks'],
            "segmentation_mask"
        )
        
        print(f"\nProcessing complete!")
        print(f"- Individual masks saved as segmentation_mask_*.png")
        if combined_file:
            print(f"- Combined visualization: {combined_file}")
        
        # Print summary of the largest masks
        print("\nTop 5 largest masks:")
        sorted_masks = sorted(result['masks'], key=lambda x: x['area'], reverse=True)
        for i, mask_data in enumerate(sorted_masks[:5]):
            print(f"{i+1}. Area: {mask_data['area']:,} pixels, IoU: {mask_data['predicted_iou']:.3f}")
    else:
        print("Failed to generate masks")