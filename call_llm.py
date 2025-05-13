import openai

client = openai.OpenAI()

SYSTEM_MSG = """
You are an expert Python robotics engineer.

Output STRICTLY Python code — no markdown, comments outside code, or prose.

Global constraints
------------------
• Edit ONLY inside the TODO regions.
• Use ONLY the helper primitives listed.
• Imports are already present; do not add new ones.
• All literals must be SI units (metres, radians, seconds).
"""


USER_MSG = """

You are given a tak from the user that they want the robot to perform.
Task: {task}
Your objective is to use a set of primtives available to you to write code that get the robot to perform this task.


# ------------- start of primitive list -------------


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

go_rest(bot: InterbotixManipulatorXS) -> None:
    /"/"/"
    Move the robot to rest position.
    
    This function moves the robot arm to its rest position and waits
    for the operation to complete.
    
    Args:
        bot: InterbotixManipulatorXS object
    /"/"/"
    
visual_qa(image: Image.Image, prompt: str) -> str:
    /"/"/"
    Query a Vision Language Model (VLM) with an image and prompt.
    
    This function sends an image and text prompt to GPT-4o and returns
    the model's response.
    
    Args:
        image: Image to send to the VLM
        prompt: Text prompt to send with the image
    
    Returns:
        The VLM's text response
    /"/"/"


# ------------- end of primitive list -------------

Using these primitives, you need to implement a function that performs the given task.
The function should be called `control_robot` and should have the following signature:
def control_robot(bot: InterbotixManipulatorXS):
    ...

    
Here is an example where the task was: "pick up the green alien"

# ---------- start of example ------
def control_robot(bot: InterbotixManipulatorXS):                                                                                                      
    image, depth_image = capture_scene_data()                                                                                                         
    masks, keypoints = generate_keypoints(image)                                                                                                      
    alien_point = select_keypoint(image, masks, keypoints, "green alien")                                                                             
    alien_3d_point = project_to_3d(alien_point, depth_image)                                                                                          
                                                                                                                                                      
    set_gripper_pose(bot, alien_3d_point[0], alien_3d_point[1], alien_3d_point[2] + 0.1, 0, 1.5708, 0)                                                
    set_gripper_pose(bot, alien_3d_point[0], alien_3d_point[1], alien_3d_point[2] + 0.01, 0, 1.5708, 0)                                               
    close_gripper(bot)                                                                                                                                
    go_home(bot)   
# ---------- end of example ----------


Guidelines:
- When picking up objects, point the gripper downwards by using +90 degrees of pitch.
- When picking up objects, first approach a point above the object (0.1m above).
- When picking up objects, target slightly below the point (-0.01m) before closing the gripper.
- Positive z is up, negative z is down.
- Use the visual_qa function to learn images
- Use capture_scene_data to get the image of the scene at required points in time
- If you need to interact with multiple objects, first use the visual_qa function to get a list of objects in the scene (the function will return a string but you can ask it for a list and parse it) satistying a set of conditions. This can be followed by iterating over them and taking the required operations.
- If you are asking the VLM something, make sure to tell it the format you need the answer it specifically. For example if you want a comma seperated list, you should specify this in the prompt. Be as specific as possible. 
- Going to the sleep pose is a good way to observe and inspect objects the gripper has picked up.
- Remember to drop objects somewhere after inspecting them.
- Going to the home pose and opening the gripper is a good way to hand the object to a human or to clear it from the table.

Task: {task}
"""


EXAMPLE = """# ---------- start of example ----------

Task: "Approach, grasp and lift an item at p3d using a safe vertical trajectory."

Code:
def control robot(bot: InterbotixManipulatorXS, p3d: np.ndarray) -> None:
    \"\"\"Approach, grasp and lift an item at p3d using a safe vertical trajectory.\"\"\"
    x, y, z = p3d
    APPROACH_Z = 0.10          # approach clearance [m]
    set_gripper_pose(bot, x, y, z + APPROACH_Z, roll=0, pitch=1.5708, yaw=0)
    move_by_offset(bot, dz=-APPROACH_Z)
    close_gripper(bot)
    move_by_offset(bot, dz=APPROACH_Z * 2)

    
# ---------- end of example ----------
"""

def code_for_request(prompt: str) -> str:

    resp = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": USER_MSG.format(task=prompt)},
        ],
        # temperature=0
    )

    generated_code = resp.choices[0].message.content

    return generated_code
