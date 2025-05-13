import openai

client = openai.OpenAI()

TASK_DESCRIPTION = "Acquire RGB-D data, choose a grasp point on the green alien, map it to 3-D, and command the robot to pick it up."

SYSTEM_MSG = """
You are an expert Python robotics engineer.

Output STRICTLY Python code — no markdown, comments outside code, or prose.

Global constraints
------------------
• Edit ONLY inside the TODO regions.
• Use ONLY the helper primitives listed.
• Imports are already present; do not add new ones.
• All literals must be SI units (metres, radians, seconds).

Safety checklist (must be obvious in the code)
----------------------------------------------
1. Define or reuse APPROACH_Z = 0.10  # m (vertical clearance before descend).
2. Descend exactly APPROACH_Z, close gripper, then RETREAT ≥ 2×APPROACH_Z.
3. End-effector orientation: roll = 0, pitch = +π/2, yaw = 0.
4. Always call go_home(bot) in a finally-block on any raised exception.

SELECTION_PROMPT rules
----------------------
• Derive from the control_robot docstring.
• ≤ 50 words, must reference the target object (e.g. “green alien”).
• Must instruct the VLM to “return ONLY the point”.
• Must end with a period.
"""


USER_MSG = f"""
Below is an EXAMPLE function that shows the preferred style.

# ---------- EXAMPLE ----------
def pick_up_from_point(bot: InterbotixManipulatorXS, p3d: np.ndarray) -> None:
    \"\"\"Approach, grasp and lift an item at p3d using a safe vertical trajectory.\"\"\"
    x, y, z = p3d
    APPROACH_Z = 0.10          # approach clearance [m]
    set_gripper_pose(bot, x, y, z + APPROACH_Z, roll=0, pitch=1.5708, yaw=0)
    move_by_offset(bot, dz=-APPROACH_Z)
    close_gripper(bot)
    move_by_offset(bot, dz=APPROACH_Z * 2)

# ------------- helper primitives you MAY call -------------
query_vlm(image_path: Path, prompt: str)  -> str : 
    /"/"/"
    Query a Vision Language Model (VLM) with an image and prompt.
    
    This function sends an image and text prompt to GPT-4o and returns
    the model's response.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt to send with the image
    
    Returns:
        The VLM's text response
    /"/"/"

capture_scene_data(directory: Path = IMAGES_DIR) -> Tuple[Image.Image, np.ndarray] :
    /"/"/"
    Capture image and depth data from the camera and save them.
    
    This function captures an RGB image and a depth image from the
    RealSense camera, saves them to disk, and returns them.
    
    Args:
        directory: Directory to save the captured images
    
    Returns:
        Tuple of (PIL Image, depth image as numpy array)
    /"/"/"

generate_keypoints(image: Image.Image,model_name: str = MODEL_NAME, seed: int = 42) -> Tuple[List[Dict[str, Any]], np.ndarray]: 
    /"/"/"
    Generate keypoints from mask generation.
    
    This function generates masks from an image using the specified model,
    then samples points from each mask to create keypoints.
    
    Args:   
        image: Source image
        model_name: Name of the model to use
        seed: Random seed for point sampling
    
    Returns:
        Tuple of (masks, keypoints) where masks is a list of dictionaries       
    /"/"/"

select_keypoint(image: Image.Image, masks: List[Dict[str, Any]], points: np.ndarray, prompt: str) -> np.ndarray :
    /"/"/"
    Select a keypoint from a list of masks and points.
    
    This function selects a keypoint from a list of masks and points based on a prompt.
    
    Args:
        image: Source image
        masks: List of masks
        points: List of points
        prompt: Prompt to select a keypoint

    Returns:
        Selected (x,y) point as a numpy array
    /"/"/"

project_to_3d(pixel_point: Tuple[int, int], depth_image: np.ndarray) -> np.ndarray :
    /"/"/"
    Get 3D point from pixel coordinate and transform it.
    
    This function converts a 2D pixel coordinate to a 3D point using
    the depth image, then transforms it from the camera frame to the
    robot frame.
    
    Args:
        pixel_point: (x, y) pixel coordinate
        depth_image: Depth image as numpy array
    
    Returns:
        3D point in robot frame as numpy array [x, y, z]
    
    Raises:
        Point3DException: If 3D point conversion fails
    /"/"/"

sleep(seconds: float) -> None :
    /"/"/"
    Pause execution for a specified number of seconds.
    
    This function pauses the execution of the program for a specified
    number of seconds.
    
    Args:
        seconds: Number of seconds to pause
    
    Returns:
        None
    /"/"/"

open_gripper(bot: InterbotixManipulatorXS) -> None :
    /"/"/"
    Open the robot gripper.
    
    This function opens the robot gripper and waits for the operation
    to complete.
    
    Args:
        bot: InterbotixManipulatorXS object
    /"/"/"

close_gripper(bot: InterbotixManipulatorXS) -> None :
    /"/"/"
    Close the robot gripper.
    
    This function closes the robot gripper and waits for the operation
    to complete.
    /"/"/"

go_home(bot: InterbotixManipulatorXS) -> None :
    /"/"/"
    Move the robot to the home position.
    
    This function moves the robot to the home position and waits for the operation
    to complete.
    /"/"/"

move_to_point(bot: InterbotixManipulatorXS, x: float, y: float, z: float) -> None :
    /"/"/"
    Move the robot arm to a 3D point in the robot frame
    
    Args:
        bot: InterbotixManipulatorXS object
        x: X coordinate in robot frame
        y: Y coordinate in robot frame
        z: Z coordinate in robot frame
        
    Raises:
        RobotOperationException: If the movement fails
    /"/"/"

move_by_offset(bot: InterbotixManipulatorXS, dx: float = 0, dy: float = 0, dz: float = 0) -> None :
    /"/"/"
    Move the robot arm by a specified offset.
    
    This function moves the robot arm by the specified offset from
    its current position.
    
    Args:
        bot: InterbotixManipulatorXS object
        dx: Offset in x direction (meters)
        dy: Offset in y direction (meters)
        dz: Offset in z direction (meters)
    /"/"/"

rotate_by_offset(bot: InterbotixManipulatorXS, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None :
    /"/"/"
    Rotate the gripper by a specified offset.
    
    This function rotates the robot gripper by the specified Euler angle
    offsets from its current orientation.
    
    Args:
        bot: InterbotixManipulatorXS object
        roll: Roll angle offset in radians
        pitch: Pitch angle offset in radians
        yaw: Yaw angle offset in radians
    /"/"/"

set_gripper_pose(bot: InterbotixManipulatorXS, x: float, y: float, z: float, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None :
    /"/"/"
    Set the gripper pose using position and orientation.
    
    This function sets the position and orientation of the robot gripper
    in the robot frame.
    
    Args:
        bot: InterbotixManipulatorXS object
        x: X coordinate in robot frame
        y: Y coordinate in robot frame
        z: Z coordinate in robot frame
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
    /"/"/"
    
# ---------- TASK ----------
Implement BOTH of these TODOs in robot_control.py

1) Write SELECTION_PROMPT that asks the VLM for exactly one (x,y) pixel
   on the green alien’s body, following the rules in SYSTEM_MSG.

2) Implement control_robot(bot) so that it:
   • captures an RGB-D frame,
   • generates candidate keypoints,
   • queries the VLM with SELECTION_PROMPT,
   • projects the chosen pixel to 3-D,
   • performs the same approach-grasp-retreat pattern as in the example,
   • returns the arm to a safe home pose in a finally-block.

Use the constant TASK_DESCRIPTION = \"{TASK_DESCRIPTION}\" in the docstring.
Do NOT change any code outside the TODO regions.
"""


resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": USER_MSG}
    ],
    temperature=0.1  # deterministic code
)

generated_code = resp.choices[0].message.content

print(generated_code)