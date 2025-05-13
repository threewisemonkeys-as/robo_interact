import os
import time
import torch
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the mask generator
mask_generator = None
CUR_DIR = Path(__file__).parent

def get_sam_mask_generator(name: str, sam_checkpoint: str):
    """Initialize and return SAM mask generator"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    model_start = time.time()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    model_type = "vit_b"
    
    # Initialize the SAM model using the registry
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
    
    # Initialize mask generator
    mask_start = time.time()
    logger.info("Initializing mask generator")
    generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=0.95,
    )
    logger.info(f"Mask generator initialized in {time.time() - model_start:.2f} seconds")
    return generator

def serialize_masks(masks):
    """Convert masks to JSON-serializable format"""
    serialized_masks = []
    for mask in masks:
        # Convert numpy arrays to lists and ensure all data is JSON-serializable
        serialized_mask = {}
        for key, value in mask.items():
            if isinstance(value, np.ndarray):
                if value.dtype == bool:
                    # Convert boolean mask to base64 encoded string
                    mask_uint8 = value.astype(np.uint8) * 255
                    mask_bytes = mask_uint8.tobytes()
                    serialized_mask[key] = {
                        'data': base64.b64encode(mask_bytes).decode(),
                        'shape': value.shape,
                        'dtype': 'bool'
                    }
                else:
                    serialized_mask[key] = value.tolist()
            else:
                serialized_mask[key] = value
        serialized_masks.append(serialized_mask)
    return serialized_masks

@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    try:
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate masks
        logger.info("Generating masks - this may take some time")
        mask_start = time.time()
        masks = mask_generator.generate(np.array(image.convert("RGB")))
        logger.info(f"Generated {len(masks)} masks in {time.time() - mask_start:.2f} seconds")
        
        # Serialize masks for JSON response
        serialized_masks = serialize_masks(masks)
        
        return jsonify({
            'status': 'success',
            'num_masks': len(masks),
            'masks': serialized_masks,
            'generation_time': time.time() - mask_start
        })
        
    except Exception as e:
        logger.error(f"Error generating masks: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': mask_generator is not None})

if __name__ == '__main__':
    # Initialize the mask generator once when the server starts
    sam_checkpoint = sys.argv[1]
    logger.info("Loading SAM model...")
    mask_generator = get_sam_mask_generator("sam_vit_b", sam_checkpoint)
    logger.info("Server ready to accept requests")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)