import time
import logging
import random
from pathlib import Path
import base64
import mimetypes

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set this before importing pyplot
import matplotlib.pyplot as plt

import torch
from PIL import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import openai

import rospy
from sensor_msgs.msg import Image as ROSSensorImage 
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray


CUR_DIR = Path(__file__).resolve().parent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force override any existing configuration
)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Get logger and add handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Make sure logs will display
logger.propagate = True

# Test log message
logger.info("Logging system initialized")


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


def get_sam_mask_generator(
    name: str
):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    model_start = time.time()

    if name == "sam":
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam_checkpoint: str = CUR_DIR / "3p/sam_ckpts/sam_vit_h_4b8939.pth"
        model_type: str = "vit_h"

        # Initialize the SAM model using the registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        
        # Mask generation
        mask_start = time.time()
        logger.info("Initializing mask generator")
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.95,
        )
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")

    elif name == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2_checkpoint = CUR_DIR / "3p/sam2_ckpts/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        
        # Mask generation
        mask_start = time.time()
        logger.info("Initializing mask generator")
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        

    else:
        raise ValueError(f"Unknown model name: {name}")
    

    return mask_generator


def get_3d_point_from_pixel(pixel_coord):
    """
    Convert a 2D pixel coordinate to a 3D point using the RealSense depth camera.
    
    Args:
        pixel_coord (tuple or np.ndarray): (x, y) pixel coordinate
        
    Returns:
        np.ndarray: 3D point in camera frame (x, y, z) in meters
    """
    # Extract pixel coordinates
    x, y = int(pixel_coord[0]), int(pixel_coord[1])
    
    # Make sure ROS node is initialized
    if not rospy.core.is_initialized():
        logger.info("Initializing ROS node for 3D point conversion")
        rospy.init_node('pixel_to_3d_converter', anonymous=True)
        logger.info("ROS node initialized")
    
    try:
        # Get depth image
        logger.info(f"Waiting for depth image on topic '/camera/aligned_depth_to_color/image_raw'")
        depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', ROSSensorImage, timeout=2.0)
        logger.info(f"Received depth image")
        
        # Convert ROS Image message to numpy array
        from cv_bridge import CvBridge
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        
        # Get camera intrinsics
        logger.info(f"Waiting for camera info on topic '/camera/aligned_depth_to_color/camera_info'")
        camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo, timeout=2.0)
        logger.info(f"Received camera info")
        
        # Create RealSense intrinsics object
        intrinsics = rs.intrinsics()
        intrinsics.width = camera_info_msg.width
        intrinsics.height = camera_info_msg.height
        intrinsics.ppx = camera_info_msg.K[2]  # Principal point x
        intrinsics.ppy = camera_info_msg.K[5]  # Principal point y
        intrinsics.fx = camera_info_msg.K[0]   # Focal length x
        intrinsics.fy = camera_info_msg.K[4]   # Focal length y
        
        # Set the distortion model (RealSense uses Brown-Conrady model)
        intrinsics.model = rs.distortion.brown_conrady
        intrinsics.coeffs = list(camera_info_msg.D[:5])  # RealSense uses 5 distortion coefficients
        
        # Get depth value at the specified pixel (in millimeters)
        depth_mm = depth_image[y, x]
        
        # Check if depth is valid
        if depth_mm <= 0 or np.isnan(depth_mm):
            logger.warning(f"Invalid depth value at pixel ({x}, {y}): {depth_mm}")
            return None
        
        # Convert depth to meters for deproject function
        depth_m = depth_mm / 1000.0
        
        # Deproject to 3D point using RealSense library
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
        
        return np.array(point_3d)
        
    except Exception as e:
        logger.error(f"Error getting 3D point: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_point_marker(point_3d, point_idx, frame_id="camera_color_optical_frame"):
    """
    Create a visualization marker for a 3D point.
    
    Args:
        point_3d (np.ndarray): 3D point coordinates [x, y, z]
        point_idx (int): Index or ID for the marker
        frame_id (str): The coordinate frame to use
        
    Returns:
        Marker: ROS visualization marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "sampled_points"
    marker.id = point_idx
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    
    # Set position
    marker.pose.position.x = point_3d[0]
    marker.pose.position.y = point_3d[1]
    marker.pose.position.z = point_3d[2]
    
    # Set orientation (identity quaternion)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    
    # Set scale
    marker.scale.x = 0.02  # 2cm sphere
    marker.scale.y = 0.02
    marker.scale.z = 0.02
    
    # Set color (lime green, matching your 2D visualization)
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    # Set lifetime (0 = forever)
    marker.lifetime = rospy.Duration(0)
    
    return marker


def create_text_marker(point_3d, point_idx, text, frame_id="camera_color_optical_frame"):
    """
    Create a text marker to label a 3D point.
    
    Args:
        point_3d (np.ndarray): 3D point coordinates [x, y, z]
        point_idx (int): Index or ID for the marker
        text (str): Text to display
        frame_id (str): The coordinate frame to use
        
    Returns:
        Marker: ROS visualization marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "point_labels"
    marker.id = point_idx
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    
    # Set position (slightly above the point)
    marker.pose.position.x = point_3d[0]
    marker.pose.position.y = point_3d[1]
    marker.pose.position.z = point_3d[2] + 0.03  # 3cm above the point
    
    # Set orientation (identity quaternion)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    
    # Set scale (text height in meters)
    marker.scale.z = 0.02
    
    # Set color (white text)
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    
    # Set text
    marker.text = text
    
    # Set lifetime (0 = forever)
    marker.lifetime = rospy.Duration(0)
    
    return marker

def query_vlm(image_path: Path, prompt: str) -> str:

    client = openai.OpenAI()

    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    image_data_url = f"data:{mime};base64,{b64}"


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                    "text": prompt},
                    {"type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        "detail": "high"
                    }}
                ]
            }
        ],
        max_tokens=250
    )

    logger.info(f"VLM response: {response}")
    return response.choices[0].message.content




MODEL_NAME = "sam"
SELECTION_PROMPT = """Which point should I grasp to pick up the string? I prefer the point on the body of the string. Only return the point, no other text."""


def main(
    selector: str = "human"
):
    if selector not in ["human", "vlm"]:
        raise ValueError(f"Unknown selector: {selector}")

    # Initialize ROS node if not already initialized
    if not rospy.core.is_initialized():
        rospy.init_node('sam_visualization', anonymous=True)
        logger.info("ROS node 'sam_visualization' initialized")
    
    # Create publishers - only for the 3D point markers
    marker_pub = rospy.Publisher('/sam_points', MarkerArray, queue_size=10)
    logger.info("ROS marker publisher created successfully")
    
    # Bridge for converting images
    bridge = CvBridge()
    
    # Open the image
    image_path = CUR_DIR / "data/images/color_0_1746999582.png"
    logger.info(f"Loading image from: {image_path}")
    
    try:
        image = Image.open(image_path)
        logger.info(f"Image loaded successfully: {image.size}x{image.mode}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return
    
    # Generate masks
    logger.info(f"Generating masks using {MODEL_NAME} model")
    try:
        masks = get_masks(MODEL_NAME, image)
        logger.info(f"Successfully generated {len(masks)} masks")
    except Exception as e:
        logger.error(f"Error generating masks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Generate sampled points for each mask
    sampled_points_per_mask = sample_points_from_masks(masks, points_per_mask=2, seed=42)
    
    # Flatten all points into a single list
    all_points = []
    for points in sampled_points_per_mask:
        all_points.extend(points)
    
    # Convert to numpy array if not empty
    if all_points:
        all_points = np.array(all_points)
    else:
        logger.error("No points were sampled from masks!")
        return
    
    # Display the image with points for selection, but save to file instead of showing
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
            ax.annotate(f"p{i}", 
                       (x+5, y+5),  # Offset position
                       color='white',
                       fontsize=12,
                       fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
    
    plt.axis('off')
    
    # Save the visualization to a file
    visualization_path = CUR_DIR / "visualization.png"
    plt.savefig(visualization_path)
    plt.close()  # Close the figure to free memory
    
    # Print points information for selection
    print("\nAvailable points:")
    for i, (x, y) in enumerate(all_points):
        print(f"Point {i}: ({x}, {y})")
    
    print(f"\nVisualization saved to: {visualization_path}")
    print("Please open this image to see all points with labels.")
    
    if selector == "human":
        # Get user selection
        chosen_point_idx = int(input("\nEnter the index of the point you want to choose: "))
        chosen_point = all_points[chosen_point_idx]
    elif selector == "vlm":
        # Use VLM to select a point
        chosen_point_response = query_vlm(
            image_path=visualization_path,
            prompt=SELECTION_PROMPT,
        )
        print(f"\nVLM response: {chosen_point_response}")
        chosen_point_idx = int(chosen_point_response[1:])
        chosen_point = all_points[chosen_point_idx]


    print(f"Chosen point: {chosen_point}")
    
    # Get depth image for 3D conversion and point cloud creation
    print("Waiting for depth image...")
    depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', ROSSensorImage, timeout=2.0)
    camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo, timeout=2.0)
    
    # Convert ROS Image message to numpy array
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    
    # Get 3D point
    logger.info(f"Getting 3D point for pixel: {chosen_point}")
    p3d = get_3d_point_from_pixel(chosen_point)
    if p3d is None:
        logger.error("Could not get 3D point. Check depth data.")
        return
    
    logger.info(f"3D point: {p3d}")
    
    # Create markers for the chosen point
    logger.info("Creating visualization markers")
    marker_array = MarkerArray()
    
    # Add sphere marker for the point
    point_marker = create_point_marker(p3d, 0)
    marker_array.markers.append(point_marker)
    
    # Add text label for the point
    text_marker = create_text_marker(p3d, 1, f"Point {chosen_point_idx}")
    marker_array.markers.append(text_marker)
    
    # Publish visualization data at a reasonable rate
    rate = rospy.Rate(10)  # 10 Hz
    
    logger.info("Publishing 3D point markers to RViz. Press Ctrl+C to stop.")
    logger.info("Make sure to set your fixed frame to 'camera_color_optical_frame' in RViz.")
    logger.info("Add a MarkerArray display in RViz with topic '/sam_points'")
    
    try:
        while not rospy.is_shutdown():
            # Update timestamps
            now = rospy.Time.now()
            for m in marker_array.markers:
                m.header.stamp = now
            
            # Publish point marker
            marker_pub.publish(marker_array)
            
            rate.sleep()
    except KeyboardInterrupt:
        logger.info("Stopped publishing visualization data")
    
    print("Publishing visualization markers to RViz. Press Ctrl+C to stop.")
    print("Make sure to set your fixed frame to 'camera_color_optical_frame' in RViz.")
    print("Add the following displays in RViz:")
    print("  1. MarkerArray with topic '/sam_points'")
    print("  2. PointCloud2 with topic '/sam_masks'")
    print("  3. Image with topic '/sam_image'")
    
    try:
        while not rospy.is_shutdown():
            # Update timestamps
            now = rospy.Time.now()
            for m in marker_array.markers:
                m.header.stamp = now
            
            # Publish point marker
            marker_pub.publish(marker_array)
            
            rate.sleep()
    except KeyboardInterrupt:
        logger.info("Stopped publishing visualization data")


if __name__ == '__main__':
    import fire
    fire.Fire(main)