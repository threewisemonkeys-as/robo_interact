#!/usr/bin/env python3
import rospy
import tf2_ros
import geometry_msgs.msg

# Initialize ROS node
rospy.init_node('transform_retriever')

# Create a TF buffer and listener
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# Wait for transform to be available
rospy.sleep(1.0)

try:
    # Get the transform from camera to arm base
    transform = tf_buffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time(0))
    
    # Print the transform
    print("Translation:")
    print(f"  X: {transform.transform.translation.x}")
    print(f"  Y: {transform.transform.translation.y}")
    print(f"  Z: {transform.transform.translation.z}")
    
    print("Rotation (Quaternion):")
    print(f"  X: {transform.transform.rotation.x}")
    print(f"  Y: {transform.transform.rotation.y}")
    print(f"  Z: {transform.transform.rotation.z}")
    print(f"  W: {transform.transform.rotation.w}")
    
except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    rospy.logerr(f"Error looking up transform: {e}")