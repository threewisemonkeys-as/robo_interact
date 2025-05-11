import matplotlib.pyplot as plt
from PIL import Image


from src.utils import show_anns
from src.mask_gen import get_masks, sample_points_from_masks

def mask_viz(
    model_name: str,
    image_path: str,
):
    image = Image.open(image_path)    

    masks = get_masks(model_name, image)
    sampled_points = sample_points_from_masks(masks, points_per_mask=5, seed=None)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(
        masks, 
        borders=True, 
        show_boxes=True, 
        show_points=True,  # Show original points (if any)
        sampled_points=sampled_points,  # Show sampled points
        sampled_point_marker='o',  # Use circle markers for sampled points
        sampled_point_size=50,  # Make them larger
        sampled_point_color='lime'  # Bright green to stand out
    )
    plt.axis('off')
    
    plt.show()



if __name__ == '__main__':

    import fire
    fire.Fire(mask_viz)