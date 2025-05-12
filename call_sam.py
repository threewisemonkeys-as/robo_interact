#!/usr/bin/env python3
"""
SAM API Client Examples
Shows how to call the SAM server API from Python
"""

import requests
import base64
import json
from PIL import Image
import io
import numpy as np

# Your ngrok URL
SERVER_URL = "https://ee30-128-84-100-13.ngrok-free.app"

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
    Segment image with point prompts
    
    Args:
        image: PIL Image object
    """

    image = encode_image_to_base64(image_path)

    data = {
        "image": image,
    }
    
    # Make API call
    response = requests.post(f"{SERVER_URL}/generate_masks", json=data)
    
    if response.status_code == 200:
        result = response.json()
        masks = []
        for mask in result['masks']:
            masks.append(decode_base64_to_image(mask))
        return masks
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

# Example usage
if __name__ == "__main__":
    # Check server status first
    print("Checking server status...")
    check_server_status()
    
    print("\nExample 1: Segmenting with points...")
    result = generate_mask(
        image_path="sample_image.jpg",
    )
    
    if result:
        mask_image = result['masks']
        mask_image.save("segmentation_mask.png")
        print("Mask saved as segmentation_mask.png")