import time
import logging
import random
from pathlib import Path
import base64
import mimetypes
import traceback
from joblib import Memory

memory = Memory('.cache', verbose=0)

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
import tf2_geometry_msgs
import tf2_ros
import geometry_msgs.msg

from interbotix_xs_modules.arm import InterbotixManipulatorXS


CUR_DIR = Path(__file__).resolve().parent

# Global configuration flags
USE_VLM = True  # Set to False to use human selection instead
MODEL_NAME = "sam"
IMAGES_DIR = CUR_DIR / "images"

# Define custom exception hierarchy
class RoboMindException(Exception):
    """Base exception for RoboMind operations."""
    pass

class ImageCaptureException(RoboMindException):
    """Exception raised when image capture fails."""
    pass

class PointGenerationException(RoboMindException):
    """Exception raised when keypoint generation fails."""
    pass

class Point3DException(RoboMindException):
    """Exception raised when 3D point conversion fails."""
    pass

class RobotOperationException(RoboMindException):
    """Exception raised when robot operation fails."""
    pass


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


def show_anns(anns, borders=True, show_boxes=True, show_points=True, show_masks=True, sampled_points=None,
sampled_point_marker='o', 
              sampled_point_size=30, sampled_point_color='green'):
    """
    Visualize segmentation masks and points.
    
    Args:
        anns (list): List of mask dictionaries
        borders (bool): Whether to show mask borders
        show_boxes (bool): Whether to show bounding boxes
        show_points (bool): Whether to show original points
        show_masks (bool): Whether to color and display the masks
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
    
    if show_masks:
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
    elif show_boxes:
        # If we're not showing masks but want to show boxes, we still need to draw them
        logger.info("Drawing bounding boxes without masks")
        for i, ann in enumerate(sorted_anns):
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Use a default color since we don't have mask colors
                rect = plt.Rectangle((x, y), w, h, linewidth=1.5, 
                                    edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
    
    if borders and show_masks:  # Only show borders if masks are enabled
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
                    # Use mask color for points if masks are shown, otherwise use a default color
                    if show_masks:
                        x, y = int(points[0][0]), int(points[0][1])
                        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                            point_color = img[y, x, :3]
                        else:
                            point_color = np.concatenate([np.random.random(3), [0.5]])[:3]  # Fallback color
                    else:
                        point_color = 'red'  # Default color when masks are not shown
                        
                    ax.scatter(points[:, 0], points[:, 1], color=point_color,
                              s=20, marker='*', edgecolors='white')
                elif hasattr(points, 'size') and points.size > 0:
                    # Already a numpy array with points
                    if show_masks:
                        x, y = int(points[0][0]), int(points[0][1])
                        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                            point_color = img[y, x, :3]
                        else:
                            point_color = np.concatenate([np.random.random(3), [0.5]])[:3]  # Fallback color
                    else:
                        point_color = 'red'  # Default color when masks are not shown
                        
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


@memory.cache
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

        sam_checkpoint: str = CUR_DIR / "3p/sam_ckpts/sam_vit_b_01ec64.pth"
        model_type: str = "vit_b"

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


def project_to_3d(pixel_point, depth_image):
    """
    Get 3D point from pixel coordinate and validate it.
    
    Raises:
        Point3DException: If 3D point conversion fails
    """
    logger.info(f"Getting 3D point for pixel: {pixel_point}")
    p3d = get_3d_point_from_pixel(pixel_point, depth_image)
    if p3d is None:
        raise Point3DException("Could not get 3D point. Check depth data.")
    
    logger.info(f"3D point: {p3d}")
    return p3d


def get_3d_point_from_pixel(pixel_coord, depth_image = None):
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
        if depth_image is None:
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


def capture_scene_data(directory=IMAGES_DIR):
    """
    Capture image and depth data from the camera and save them with timestamp filenames.
    
    Raises:
        ImageCaptureException: If image or depth data capture fails
    """
    # Create bridge for converting images
    bridge = CvBridge()
    
    # Generate timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Ensure directory exists
    directory.mkdir(parents=True, exist_ok=True)
    
    # Capture RGB image
    logger.info("Capturing live image from RealSense camera")
    image = capture_live_image(bridge)
    if image is None:
        raise ImageCaptureException("Failed to capture RGB image from camera")
    
    logger.info(f"Image captured successfully: {image.size}x{image.mode}")
    
    # Save RGB image
    image_path = directory / f"rgb_{timestamp}.png"
    image.save(image_path)
    logger.info(f"Image saved to: {image_path}")
    
    # Get depth image 
    try:
        logger.info("Waiting for depth image...")
        depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', ROSSensorImage, timeout=2.0)
        camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo, timeout=2.0)
    except rospy.ROSException as e:
        raise ImageCaptureException(f"Failed to receive depth image: {e}")
    
    # Convert ROS Image message to numpy array
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    
    # Optionally save depth image visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    depth_vis_path = directory / f"depth_{timestamp}.png"
    plt.savefig(depth_vis_path)
    plt.close()
    logger.info(f"Depth visualization saved to: {depth_vis_path}")
    
    return image, depth_image


def capture_live_image(bridge=None):
    """
    Capture a live image from the RealSense camera.
    
    Args:
        bridge (CvBridge, optional): Bridge for converting ROS images
        
    Returns:
        PIL.Image: Captured image from the camera
    """
    if bridge is None:
        bridge = CvBridge()
    
    logger.info("Waiting for color image from camera...")
    try:
        # Wait for a color image from the RealSense camera
        color_msg = rospy.wait_for_message('/camera/color/image_raw', ROSSensorImage, timeout=5.0)
        logger.info("Color image received successfully")
        
        # Convert ROS image to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        
        # Convert OpenCV image to PIL Image
        pil_image = Image.fromarray(cv_image)
        
        return pil_image
    except Exception as e:
        logger.error(f"Failed to capture live image: {e}")
        logger.error(traceback.format_exc())
        return None


def generate_keypoints(image, model_name=MODEL_NAME, seed=42):
    """
    Generate keypoints from mask generation.
    
    Raises:
        PointGenerationException: If mask generation or point sampling fails
    """
    # Generate masks
    logger.info(f"Generating masks using {model_name} model")
    try:
        masks = get_masks(model_name, image)
        logger.info(f"Successfully generated {len(masks)} masks")
    except Exception as e:
        logger.error(f"Error generating masks: {e}")
        logger.error(traceback.format_exc())
        raise PointGenerationException(f"Failed to generate masks: {e}")
    
    # Generate sampled points for each mask
    sampled_points_per_mask = sample_points_from_masks(masks, points_per_mask=2, seed=seed)
    
    # Flatten all points into a single list
    all_points = []
    for points in sampled_points_per_mask:
        all_points.extend(points)
    
    # Convert to numpy array if not empty
    if all_points:
        all_points = np.array(all_points)
    else:
        raise PointGenerationException("No points were sampled from masks")
    
    return masks, all_points


def select_keypoint(image, masks, points, prompt):
    """
    Visualize keypoints on image, save to file, and handle point selection.
    
    Args:
        image: Source image
        masks: Segmentation masks
        points: Array of keypoints
        prompt: Prompt for VLM selection
        
    Returns:
        np.ndarray: Selected (x,y) point
    """
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    
    # Show the annotations
    show_anns(
        masks,
        borders=False,
        show_boxes=False,
        show_points=False,
        show_masks=False,
        sampled_points=points,
        sampled_point_marker='o',
        sampled_point_size=50,
        sampled_point_color='lime'
    )
    
    # Add labels to each keypoint
    if len(points) > 0:
        ax = plt.gca()
        for i, (x, y) in enumerate(points):
            # Add text label with small offset from the point
            ax.annotate(f"p{i}", 
                       (x+5, y+5),
                       color='white',
                       fontsize=12,
                       fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
    
    plt.axis('off')
    
    # Save the visualization to a file
    visualization_path = IMAGES_DIR / f"viz_{timestamp}.png"
    plt.savefig(visualization_path)
    plt.close()  # Close the figure to free memory
    
    # Print points information for selection
    print("\nAvailable points:")
    for i, (x, y) in enumerate(points):
        print(f"Point {i}: ({x}, {y})")
    
    print(f"\nVisualization saved to: {visualization_path}")
    
    # Handle point selection
    if not USE_VLM:
        # Human selection mode
        chosen_point_idx = int(input("\nEnter the index of the point you want to choose: "))
    else:
        # VLM selection mode
        chosen_point_response = query_vlm(
            image_path=visualization_path,
            prompt=prompt,
        )
        print(f"\nVLM response: {chosen_point_response}")
        chosen_point_idx = int(chosen_point_response[1:])
    
    chosen_point = points[chosen_point_idx]
    print(f"Chosen point: {chosen_point}")
    return chosen_point




def sleep(seconds):
    """Simple wrapper for rospy.sleep with logging."""
    logger.info(f"Sleeping for {seconds} seconds")
    rospy.sleep(seconds)


def open_gripper(bot):
    """Open the robot gripper with logging and appropriate sleep."""
    logger.info("Opening gripper")
    bot.gripper.open()
    sleep(1.0)


def go_home(bot):
    """Move the robot to home position with logging and appropriate sleep."""
    logger.info("Moving arm to home position")
    bot.arm.go_to_home_pose()
    sleep(1.0)


def initialize_robot():
    """Initialize the robot arm and prepare it for operations."""
    bot = InterbotixManipulatorXS("wx250s", "arm", "gripper", init_node=False)
    go_home(bot)
    open_gripper(bot)
    return bot



def move_to_point(bot, p3d, camera_frame="camera_color_optical_frame", robot_frame="wx250s/base_link"):
    """
    Move the robot arm to a 3D point
    
    Args:
        bot: InterbotixManipulatorXS object
        p3d: 3D point in camera frame [x, y, z]
        camera_frame: The frame ID of the camera
        robot_frame: The frame ID of the robot base
    """
    logger.info(f"Moving to point {p3d} in {camera_frame} frame")
    
    # Transform point from camera frame to robot base frame
    try:        
        # Create a TF buffer and listener
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        
        # Wait for the transform to be available
        logger.info(f"Waiting for transform from {camera_frame} to {robot_frame}")
        sleep(1.0)  # Give time for the TF system to initialize
        
        # Create a PointStamped message for the camera point
        point_stamped = geometry_msgs.msg.PointStamped()
        point_stamped.header.frame_id = camera_frame
        point_stamped.header.stamp = rospy.Time(0)
        point_stamped.point.x = p3d[0]
        point_stamped.point.y = p3d[1]
        point_stamped.point.z = p3d[2]
        
        # Transform the point to the robot base frame
        transformed_point = tf_buffer.transform(point_stamped, robot_frame)
        
        # Extract the transformed coordinates
        x = transformed_point.point.x
        y = transformed_point.point.y
        z = transformed_point.point.z
        
        logger.info(f"Transformed point: ({x}, {y}, {z}) in {robot_frame} frame")
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        logger.error(f"Transform error: {e}")
        raise RobotOperationException(f"Transform error: {e}")
    
    try:
        logger.info(f"Moving to position: ({x}, {y}, {z})")
        bot.arm.set_ee_pose_components(
            x=x, y=y, z=z + 0.01,
        )
        sleep(2.0)  # Give time for the move to complete

    except Exception as e:
        logger.error(f"Error during arm movement: {e}")
        logger.error(traceback.format_exc())
        # Try to recover by going to home position
        go_home(bot)
        raise RobotOperationException(f"Error during arm movement: {e}")


def pick_up_from_point(bot, p3d, camera_frame="camera_color_optical_frame", robot_frame="wx250s/base_link"):
    """
    Move the robot arm to a 3D point and pick up an object.
    
    Args:
        bot: InterbotixManipulatorXS object
        p3d: 3D point in camera frame [x, y, z]
        camera_frame: The frame ID of the camera
        robot_frame: The frame ID of the robot base
    """
    logger.info(f"Moving to point {p3d} in {camera_frame} frame")
    
    # Transform point from camera frame to robot base frame
    try:        
        # Create a TF buffer and listener
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        
        # Wait for the transform to be available
        logger.info(f"Waiting for transform from {camera_frame} to {robot_frame}")
        sleep(1.0)  # Give time for the TF system to initialize
        
        # Create a PointStamped message for the camera point
        point_stamped = geometry_msgs.msg.PointStamped()
        point_stamped.header.frame_id = camera_frame
        point_stamped.header.stamp = rospy.Time(0)
        point_stamped.point.x = p3d[0]
        point_stamped.point.y = p3d[1]
        point_stamped.point.z = p3d[2]
        
        # Transform the point to the robot base frame
        transformed_point = tf_buffer.transform(point_stamped, robot_frame)
        
        # Extract the transformed coordinates
        x = transformed_point.point.x
        y = transformed_point.point.y
        z = transformed_point.point.z
        
        logger.info(f"Transformed point: ({x}, {y}, {z}) in {robot_frame} frame")
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        logger.error(f"Transform error: {e}")
        raise RobotOperationException(f"Transform error: {e}")
    
    try:
        
        # First move to a position slightly above the target
        approach_z_offset = 0.1  # Increased for more clearance when approaching vertically
        logger.info(f"Moving to approach position: ({x}, {y}, {z + approach_z_offset})")
        
        # Set end-effector pose with downward-pointing orientation
        # Roll = 0, Pitch = 1.5708 (90 degrees), Yaw = 0
        # This makes the gripper point downward (along negative Z-axis)
        bot.arm.set_ee_pose_components(
            x=x, y=y, z=z + approach_z_offset,
            roll=0, pitch=1.5708, yaw=0
        )
        sleep(2.0)  # Give time for the move to complete
        
        # Move down to the target position
        logger.info(f"Moving to grasp position: ({x}, {y}, {z+0.05})")
        bot.arm.set_ee_cartesian_trajectory(z=-approach_z_offset)  # Move down relative to current position
        sleep(1.5)
        
        # Close the gripper to grasp the object
        logger.info("Closing gripper")
        bot.gripper.close()
        sleep(1.0)
        
        # Lift the object
        logger.info("Lifting object")
        bot.arm.set_ee_cartesian_trajectory(z=approach_z_offset*2)  # Move up relative to current position
        sleep(1.5)
        
        logger.info("Object picked up successfully")
        
    except Exception as e:
        logger.error(f"Error during arm movement: {e}")
        logger.error(traceback.format_exc())
        # Try to recover by going to home position
        go_home(bot)
        raise RobotOperationException(f"Error during arm movement: {e}")


SELECTION_PROMPT = """Which point should I grasp to pick up the green alien object? I prefer the point on the body of the object. Only return the point, no other text."""


def main():
    """Main function orchestrating the entire process."""

    # Initialize ROS node if not already initialized
    if not rospy.core.is_initialized():
        rospy.init_node('steepmind', anonymous=True)
        logger.info("ROS node 'steepmind' initialized")

    bot = initialize_robot()

    try:
        image, depth_image = capture_scene_data()
        
        masks, keypoints = generate_keypoints(image)
        chosen_keypoint = select_keypoint(image, masks, keypoints, SELECTION_PROMPT)
        p3d = project_to_3d(chosen_keypoint, depth_image)
        
        # Execute robot actions
        pick_up_from_point(bot, p3d)
        
    except Exception as e:
        logger.error(f"Attempt failed: {e}")
        logger.error(traceback.format_exc())
    
    go_home(bot)


if __name__ == '__main__':
    import fire
    fire.Fire(main)