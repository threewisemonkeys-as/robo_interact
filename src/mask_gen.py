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

def centroids_of_main_objects(
        masks,
        image_shape,                # (H, W)
        k=2,
        min_area_frac=0.01,
        border_margin=0             # set >0 if you want a buffer inside the edge
    ):
    """
    Pick the k masks that best satisfy ‘fully in view & centred’, then
    return their centroids.

    Returns
    -------
    list[dict]  each dict  { "centroid": (cx, cy),   # float
                             "area": int,
                             "distance": float,
                             "mask": <bool ndarray> }
    Raises
    ------
    ValueError  if fewer than k suitable objects are found
    """
    H, W = image_shape
    cx_img, cy_img = W / 2, H / 2

    def touches_border(bbox):
        x0, y0, w, h = bbox          # SAM’s bbox = (x, y, width, height)
        x1, y1 = x0 + w - 1, y0 + h - 1
        return (x0 <= border_margin or y0 <= border_margin or
                x1 >= W - 1 - border_margin or
                y1 >= H - 1 - border_margin)

    candidates = []
    for m in masks:
        area = m["area"] if "area" in m else m["segmentation"].sum()
        if area < min_area_frac * H * W:
            continue                             # too small

        bbox = m["bbox"] if "bbox" in m else None
        if bbox is None:
            # compute bbox quickly from the mask if SAM did not include one
            ys, xs = np.where(m["segmentation"])
            bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)

        if touches_border(bbox):
            continue                             # cropped at the edge

        # centroid
        ys, xs = np.where(m["segmentation"])
        centroid_x, centroid_y = xs.mean(), ys.mean()

        # distance to image centre
        dist = np.hypot(centroid_x - cx_img, centroid_y - cy_img)

        candidates.append((dist, -area, centroid_x, centroid_y, m))

    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} suitable object(s) found; need {k}")

    # sort: closest to centre first, larger area ties broken via -area
    candidates.sort()
    picked = candidates[:k]

    return [
        {"centroid": (cx, cy),
         "area": -neg_area,
         "distance": dist,
         "mask": mask}
        for dist, neg_area, cx, cy, mask in picked
    ]
