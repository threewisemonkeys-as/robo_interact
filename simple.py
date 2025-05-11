#!/usr/bin/env python
import rospy

if __name__ == "__main__":
    print("Before init")
    rospy.init_node('test_node', anonymous=True)
    print("After init - success!")
    rospy.spin()