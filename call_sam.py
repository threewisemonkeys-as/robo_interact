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


def generate_mask(image, SERVER_URL=SERVER_URL):
    """
    Segment image with point prompts
    
    Args:
        image: PIL Image object
    """
    data = {
        "image": image,
    }
    
    # Make API call
    response = requests.post(f"{SERVER_URL}/generate_masks", json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result
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