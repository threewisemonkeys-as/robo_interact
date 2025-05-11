from PIL import Image
import time
import logging
import numpy as np


from src.sams import get_sam_mask_generator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_masks(
    model_name: str,
    image: Image,
):
    image = np.array(image.convert("RGB"))

    sam_mask_generator = get_sam_mask_generator(model_name)
    
    logger.info("Generating masks - this may take some time")
    mask_start = time.time()
    masks = sam_mask_generator.generate(image)
    logger.info(f"Generated {len(masks)} masks in {time.time() - mask_start:.2f} seconds")
    
    return masks


def sample_points_from_masks(masks, points_per_mask=5, seed=None):
    """
    Randomly sample points from each segmentation mask.
    
    Args:
        masks (list): List of mask dictionaries returned by get_masks()
        points_per_mask (int): Number of points to sample from each mask
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        list: List of lists, where each inner list contains sampled points
              as (x, y) coordinate tuples for a mask
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    all_sampled_points = []
    
    for i, mask in enumerate(masks):
        # Get the segmentation mask
        segmentation = mask['segmentation']
        
        # Find all coordinates where the mask is True (1)
        y_coords, x_coords = np.where(segmentation)
        
        # Combine into (x, y) coordinate pairs
        valid_points = list(zip(x_coords, y_coords))
        
        # If no valid points found, continue to the next mask
        if not valid_points:
            all_sampled_points.append([])
            continue
            
        # Sample points (or take all if fewer than requested)
        if len(valid_points) <= points_per_mask:
            sampled_points = valid_points
        else:
            # Randomly sample points without replacement
            indices = np.random.choice(len(valid_points), points_per_mask, replace=False)
            sampled_points = [valid_points[i] for i in indices]
        
        all_sampled_points.append(sampled_points)
        
    return all_sampled_points