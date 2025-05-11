import matplotlib.pyplot as plt
from PIL import Image
from src.utils import show_anns, get_masks, sample_points_from_masks
import numpy as np

def mask_viz(
    model_name: str,
    image_path: str,
):
    image = Image.open(image_path)
    masks = get_masks(model_name, image)
    
    # Generate sampled points for each mask
    sampled_points_per_mask = sample_points_from_masks(masks, points_per_mask=2, seed=42)
    
    # Flatten all points into a single list
    all_points = []
    for points in sampled_points_per_mask:
        all_points.extend(points)
    
    # Convert to numpy array if not empty
    if all_points:
        all_points = np.array(all_points)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    
    # Show the annotations first
    show_anns(
        masks,
        borders=False,
        show_boxes=False,
        show_points=False,
        sampled_points=all_points,
        sampled_point_marker='o',
        sampled_point_size=50,
        sampled_point_color='lime'
    )
    
    # Add labels to each keypoint
    if len(all_points) > 0:
        ax = plt.gca()
        for i, (x, y) in enumerate(all_points):
            # Add text label with small offset from the point
            ax.annotate(f"p{i+1}", 
                       (x+5, y+5),  # Offset position
                       color='white',
                       fontsize=12,
                       fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
    
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import fire
    fire.Fire(mask_viz)

if __name__ == '__main__':
    import fire
    fire.Fire(mask_viz)