import rospy, tf2_ros, tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import pyrealsense2 as rs
import numpy as np

# assume you've already got:
u, v = clicked_pixel           # 2-D pixel from your GUI
depth_msg = rospy.wait_for_message(
    "/camera/aligned_depth_to_color/image_raw", Image)
cam_info  = rospy.wait_for_message(
    "/camera/color/camera_info", CameraInfo)

# convert the ROS CameraInfo to librealsense intrinsics
K = np.array(cam_info.K).reshape(3, 3)
intr = rs.intrinsics()
intr.width, intr.height = cam_info.width, cam_info.height
intr.ppx, intr.ppy = K[0, 2], K[1, 2]
intr.fx,  intr.fy  = K[0, 0], K[1, 1]
intr.model = rs.distortion.none
intr.coeffs = [0, 0, 0, 0, 0]

# depth value in metres (convert from uint16 mm)
depth = np.asarray(bytearray(depth_msg.data)).view(
            np.uint16).reshape(depth_msg.height, depth_msg.width)[v, u] / 1000.0

xyz_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)  # metres

