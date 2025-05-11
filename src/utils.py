import time
import logging

import numpy as np
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def show_anns(anns, borders=True, show_boxes=True, show_points=True, sampled_points=None,
sampled_point_marker='o', 
              sampled_point_size=30, sampled_point_color='green'):
    """
    Visualize segmentation masks and points.
    
    Args:
        anns (list): List of mask dictionaries
        borders (bool): Whether to show mask borders
        show_boxes (bool): Whether to show bounding boxes
        show_points (bool): Whether to show original points
        sampled_points (list | None): Sampled points
        sampled_point_marker (str): Marker style for sampled points
        sampled_point_size (int): Size of sampled points
        sampled_point_color (str or tuple): Color of sampled points
    """
    logger.info(f"Starting visualization of {len(anns)} annotations")
    start_time = time.time()
    
    if len(anns) == 0:
        logger.warning("No annotations to visualize")
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    logger.info(f"Processing {len(sorted_anns)} masks for visualization")
    mask_start = time.time()
    for i, ann in enumerate(sorted_anns):
        if i % 50 == 0 and i > 0:  # Log progress every 50 masks
            logger.info(f"Processed {i}/{len(sorted_anns)} masks")
        
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        
        # Draw bounding box
        if show_boxes and 'bbox' in ann:
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=1.5, 
                                 edgecolor=color_mask[:3], facecolor='none')
            ax.add_patch(rect)
    
    logger.info(f"Mask coloring completed in {time.time() - mask_start:.2f} seconds")
    
    if borders:
        logger.info("Adding contour borders to visualization")
        border_start = time.time()
        import cv2
        for i, ann in enumerate(sorted_anns):
            if i % 50 == 0 and i > 0:  # Log progress every 50 contours
                logger.info(f"Processed {i}/{len(sorted_anns)} contours")
                
            m = ann['segmentation']
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
        logger.info(f"Border drawing completed in {time.time() - border_start:.2f} seconds")
    
    # Draw original point coordinates
    if show_points:
        points_start = time.time()
        logger.info("Adding original point coordinates to visualization")
        for i, ann in enumerate(sorted_anns):
            if 'point_coords' in ann and ann['point_coords'] is not None:
                points = ann['point_coords']
                # Handle points as list or numpy array
                if isinstance(points, list) and len(points) > 0:
                    # Convert list to numpy array if needed
                    points = np.array(points)
                    # Use mask color for points
                    x, y = int(points[0][0]), int(points[0][1])
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        point_color = img[y, x, :3]
                    else:
                        point_color = color_mask[:3]  # Fallback color
                    ax.scatter(points[:, 0], points[:, 1], color=point_color,
                              s=20, marker='*', edgecolors='white')
                elif hasattr(points, 'size') and points.size > 0:
                    # Already a numpy array with points
                    x, y = int(points[0][0]), int(points[0][1])
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        point_color = img[y, x, :3]
                    else:
                        point_color = color_mask[:3]  # Fallback color
                    ax.scatter(points[:, 0], points[:, 1], color=point_color,
                              s=20, marker='*', edgecolors='white')
        logger.info(f"Original point visualization completed in {time.time() - points_start:.2f} seconds")
    
    # Draw sampled points
    if sampled_points is not None:
        sampled_points_start = time.time()
        logger.info("Adding sampled points to visualization")
        for i, ann in enumerate(sorted_anns):
            points = sampled_points
            # Handle points as list or numpy array
            if isinstance(points, list) and len(points) > 0:
                # Convert list to numpy array if needed
                points = np.array(points)
                ax.scatter(points[:, 0], points[:, 1], 
                            color=sampled_point_color,
                            s=sampled_point_size, 
                            marker=sampled_point_marker, 
                            edgecolors='white',
                            alpha=0.8,
                            zorder=10)  # Higher zorder to appear on top
            elif hasattr(points, 'size') and points.size > 0:
                ax.scatter(points[:, 0], points[:, 1], 
                            color=sampled_point_color,
                            s=sampled_point_size, 
                            marker=sampled_point_marker, 
                            edgecolors='white',
                            alpha=0.8,
                            zorder=10)
        logger.info(f"Sampled point visualization completed in {time.time() - sampled_points_start:.2f} seconds")
    
    ax.imshow(img)
    logger.info(f"Visualization completed in {time.time() - start_time:.2f} seconds")