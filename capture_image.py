#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
import os
import numpy as np
from PIL import Image as PILImage

class SimpleImageCapture:
    def __init__(self):
        # Initialize ROS node
        print("Initializing ROS node...")
        rospy.init_node('simple_realsense_capture', anonymous=True)
        rospy.loginfo("Simple RealSense Image Capture node initialized")
        print("ROS node initialized")

        # Directory to save images
        self.save_dir = os.path.expanduser("data/images")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Flags to track if we've received images this capture
        self.color_received = False
        self.depth_received = False
        
        # Service to trigger image capture
        rospy.Service('capture_simple_image', Empty, self.handle_capture)
        
        # Counter for unique filenames
        self.counter = 0
        
        rospy.loginfo("Simple RealSense Image Capture node started")
        rospy.loginfo("Call the 'capture_simple_image' service to capture color + depth images")
    
    def handle_capture(self, req):
        """Service handler to capture one color + depth pair"""
        # Reset flags
        self.color_received = False
        self.depth_received = False
        
        # Assign IDs/timestamps for this capture
        self.capture_id = self.counter
        self.capture_timestamp = rospy.Time.now().to_sec()
        
        # Subscribe to color and aligned depth topics
        color_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image,
            self.color_callback, queue_size=1
        )
        depth_sub = rospy.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image,
            self.depth_callback, queue_size=1
        )
        
        # Wait until both images arrive or timeout
        timeout = rospy.Duration(5.0)
        start = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz check
        
        while not (self.color_received and self.depth_received) and (rospy.Time.now() - start) < timeout:
            rate.sleep()
        
        # Unsubscribe
        color_sub.unregister()
        depth_sub.unregister()
        
        # Check results
        if self.color_received and self.depth_received:
            rospy.loginfo("Color + depth captured and saved")
            # Increment counter after successful save
            self.counter += 1
        else:
            rospy.logerr("Failed to capture both images within timeout")
        
        return EmptyResponse()
    
    def color_callback(self, data):
        """Save incoming color image as PNG"""
        if self.color_received:
            return  # already got it
        
        try:
            # Build filename
            fname = os.path.join(
                self.save_dir,
                f"color_{self.capture_id}_{self.capture_timestamp:.0f}.png"
            )
            # Convert raw bytes to H×W×3 array
            if data.encoding in ("rgb8", "bgr8"):
                arr = np.frombuffer(data.data, dtype=np.uint8).reshape(
                    data.height, data.width, 3
                )
                # BGR→RGB if needed
                if data.encoding == "bgr8":
                    arr = arr[:, :, ::-1]
                # Save via PIL
                PILImage.fromarray(arr).save(fname)
                rospy.loginfo(f"Saved color image to {fname}")
                self.color_received = True
            else:
                rospy.logerr(f"Unsupported color encoding: {data.encoding}")
        except Exception as e:
            rospy.logerr(f"Error saving color image: {e}")
    
    def depth_callback(self, data):
        """Save incoming depth image as 16-bit PNG (depth in mm)"""
        if self.depth_received:
            return  # already got it
        
        try:
            # Build filename
            fname = os.path.join(
                self.save_dir,
                f"depth_{self.capture_id}_{self.capture_timestamp:.0f}.png"
            )
            # Decode based on encoding
            if data.encoding == "16UC1":
                depth_arr = np.frombuffer(data.data, dtype=np.uint16).reshape(
                    data.height, data.width
                )
            elif data.encoding == "32FC1":
                # float32 in meters → uint16 in mm
                depth_f = np.frombuffer(data.data, dtype=np.float32).reshape(
                    data.height, data.width
                )
                depth_arr = (depth_f * 1000.0).astype(np.uint16)
            else:
                rospy.logerr(f"Unsupported depth encoding: {data.encoding}")
                return
            
            # Save as 16-bit PNG
            PILImage.fromarray(depth_arr).save(fname)
            rospy.loginfo(f"Saved depth image to {fname}")
            self.depth_received = True
        except Exception as e:
            rospy.logerr(f"Error saving depth image: {e}")

if __name__ == "__main__":
    try:
        capture_node = SimpleImageCapture()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
