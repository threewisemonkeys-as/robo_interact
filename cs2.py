import requests
import base64
import numpy as np
from PIL import Image
import io
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMClient:
    def __init__(self, server_url=' https://2cc9-128-84-100-13.ngrok-free.app'):
        self.server_url = server_url
        
    def check_server_health(self):
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking server health: {e}")
            return None
    
    def deserialize_masks(self, serialized_masks):
        """Convert serialized masks back to original format"""
        masks = []
        for serialized_mask in serialized_masks:
            mask = {}
            for key, value in serialized_mask.items():
                if isinstance(value, dict) and 'data' in value:
                    # Deserialize numpy array
                    mask_bytes = base64.b64decode(value['data'])
                    shape = tuple(value['shape'])
                    dtype = value['dtype']
                    
                    if dtype == 'bool':
                        # Convert back to bool array
                        mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(shape)
                        mask[key] = (mask_uint8 > 0).astype(bool)
                    else:
                        mask[key] = np.frombuffer(mask_bytes, dtype=np.float32).reshape(shape)
                elif isinstance(value, list):
                    # Convert list back to numpy array
                    mask[key] = np.array(value)
                else:
                    mask[key] = value
            masks.append(mask)
        return masks
    
    def get_masks(self, image_path):
        """
        Send image to server and get masks
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of masks in the same format as the original get_masks function
        """
        try:
            # Check server health first
            health = self.check_server_health()
            if not health or not health.get('model_loaded'):
                raise Exception("Server is not ready or model is not loaded")
            
            # Read image file
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                
                logger.info(f"Sending image {image_path} to server...")
                start_time = time.time()
                
                # Send POST request to server
                response = requests.post(f"{self.server_url}/generate_masks", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Received {result['num_masks']} masks from server")
                    logger.info(f"Total time (including network): {time.time() - start_time:.2f} seconds")
                    
                    # Deserialize masks
                    masks = self.deserialize_masks(result['masks'])
                    return masks
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    raise Exception(f"Server error: {error_msg}")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting masks: {e}")
            raise
    
    def get_masks_from_pil_image(self, image):
        """
        Send PIL Image to server and get masks
        
        Args:
            image (PIL.Image): PIL Image object
            
        Returns:
            list: List of masks in the same format as the original get_masks function
        """
        try:
            # Check server health first
            health = self.check_server_health()
            if not health or not health.get('model_loaded'):
                raise Exception("Server is not ready or model is not loaded")
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            files = {'image': ('image.png', img_byte_arr, 'image/png')}
            
            logger.info("Sending PIL image to server...")
            start_time = time.time()
            
            # Send POST request to server
            response = requests.post(f"{self.server_url}/generate_masks", files=files)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received {result['num_masks']} masks from server")
                logger.info(f"Total time (including network): {time.time() - start_time:.2f} seconds")
                
                # Deserialize masks
                masks = self.deserialize_masks(result['masks'])
                return masks
            else:
                error_msg = response.json().get('error', 'Unknown error')
                raise Exception(f"Server error: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting masks: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = SAMClient(" https://2cc9-128-84-100-13.ngrok-free.app")
    
    # Example 1: Using file path
    image_path = "path/to/your/image.jpg"
    try:
        masks = client.get_masks(image_path)
        print(f"Got {len(masks)} masks")
        
        # Access the first mask as an example
        if masks:
            first_mask = masks[0]
            print(f"First mask keys: {first_mask.keys()}")
            if 'segmentation' in first_mask:
                print(f"Segmentation shape: {first_mask['segmentation'].shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using PIL Image
    try:
        image = Image.open(image_path)
        masks = client.get_masks_from_pil_image(image)
        print(f"Got {len(masks)} masks from PIL image")
    except Exception as e:
        print(f"Error: {e}")