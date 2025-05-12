from interbotix_xs_modules.arm import InterbotixManipulatorXS

# This script makes the end-effector go to a specific pose by defining the pose components
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python ee_pose_components.py  # python3 bartender.py if using ROS Noetic'

def main():
    bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
    bot.gripper.open()
    bot.arm.go_to_home_pose()
    bot.gripper.close()
    bot.arm.go_to_sleep_pose()



if __name__=='__main__':
    main()
