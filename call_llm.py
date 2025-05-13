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
query_vlm(image_path: Path, prompt: str)  -> str
capture_scene_data(directory: Path = IMAGES_DIR) -> Tuple[Image.Image, np.ndarray]
generate_keypoints(image: Image.Image,model_name: str = MODEL_NAME, seed: int = 42) -> Tuple[List[Dict[str, Any]], np.ndarray]
select_keypoint(image: Image.Image, masks: List[Dict[str, Any]], points: np.ndarray, prompt: str) -> np.ndarray
project_to_3d(pixel_point: Tuple[int, int], depth_image: np.ndarray) -> np.ndarray
sleep(seconds: float) -> None
open_gripper(bot: InterbotixManipulatorXS) -> None
close_gripper(bot: InterbotixManipulatorXS) -> None
go_home(bot: InterbotixManipulatorXS) -> None
move_to_point(bot: InterbotixManipulatorXS, x: float, y: float, z: float) -> None
move_by_offset(bot: InterbotixManipulatorXS, dx: float = 0, dy: float = 0, dz: float = 0) -> None
rotate_by_offset(bot: InterbotixManipulatorXS, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None
set_gripper_pose(bot: InterbotixManipulatorXS, x: float, y: float, z: float, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None
logger -> logging.Logger

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