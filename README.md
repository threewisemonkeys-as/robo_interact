# Robot Interact

Launch arm


```
roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s
```

Launch camera and transform -


```
roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s use_pointcloud_tuner_gui:=true use_armtag_tuner_gui:=true
```


## Examples of generated code -



### Pick up the green alien

```python

def control_robot(bot):
    image, depth_image = capture_scene_data()                                                                                    
    masks, keypoints = generate_keypoints(image)                                                                                 
    green_alien_point = select_keypoint(image, masks, keypoints, "green alien figure")                                                  
    position_3d = project_to_3d(green_alien_point, depth_image)                                                                  
                                                                                                                                    
    x, y, z = position_3d                                                                                                        
    z += 0.1  # Move slightly above the ground                                                                                      
    set_gripper_pose(bot, x, y, z, roll=0, pitch=1.5708, yaw=0)  # Point gripper downwards                                                                                                                         
    z -= 0.12  # Move down to the ground level                                                                                    
    set_gripper_pose(bot, x, y, z, roll=0, pitch=1.5708, yaw=0)  # Adjust gripper position                                       
    close_gripper(bot)   
    move_by_offset(bot, dz=0.1)  # Lift the object                              
    go_home(bot)
    open_gripper(bot)  # Release the object

```


### Check which object is squishy


```python
def control_robot(bot: InterbotixManipulatorXS):
    # TODO: pick up objects one by one, inspect squishiness, and hand the squishy one to a human.                                
    image, depth_image = capture_scene_data()                                                                                    
    # Ask VLM for a comma separated list of all objects present in the scene.                                                    
    objects_str = visual_qa(image, "List all objects visible in the scene as a comma separated list with names only. Ignore the robot arm and only list objects that can be picked up")                                                                                                                              
    objects = [obj.strip() for obj in objects_str.split(",") if obj.strip()]                                                     
                                                                                                                                 
    for obj in objects:                                                                                                          
        # Refresh scene data and keypoints for the current scene.                                                                
        image, depth_image = capture_scene_data()                                                                                
        masks, keypoints = generate_keypoints(image)                                                                             
        # Select the object's keypoint.                                                                                          
        pixel_point = select_keypoint(image, masks, keypoints, f"select the {obj}")                                              
        # Project the selected 2D pixel to a 3D point in the robot frame.                                                        
        obj_3d_point = project_to_3d((int(pixel_point[0]), int(pixel_point[1])), depth_image)        
                                                                
        # Approach 0.1m above the object with the gripper pointed downwards (+90 deg pitch).                                     
        set_gripper_pose(bot, obj_3d_point[0], obj_3d_point[1], obj_3d_point[2] + 0.1, 0, 1.5708, 0)
        # Approach closer to the object.
        set_gripper_pose(bot, obj_3d_point[0], obj_3d_point[1], obj_3d_point[2] -0.01, 0, 1.5708, 0)                                             
        # Close the gripper to grasp the object.                                                                                 
        close_gripper(bot)
        go_rest(bot)                                                                      
                                                                
        # Inspect the grasped object using visual QA.                                                                            
        carried_image, _ = capture_scene_data()                                                                                  
        response = visual_qa(carried_image, "Is the object currently held by the gripper squishy? Answer with a single word: yes or no.")
         
        if "yes" in response.lower():
            # Squishy object found. Hand it over by moving to home pose.
            go_home(bot)
            break
        else:
            # Not squishy. Drop the object safely.
            open_gripper(bot)
```



### Clear all objects from the table until it is empty



```python
def control_robot(bot: InterbotixManipulatorXS):
    while True:                                                                                                                                                    
        image, depth_image = capture_scene_data()                                                                                                                  
        # Ask the VLM for a list of objects on the table.                                                                                                          
        objects_response = visual_qa(image, "Provide a comma separated list of object names on the table. If the table is clear, reply with 'clear'.")             
        objects_response = objects_response.strip().lower()                                                                                                        
        if objects_response == "clear" or objects_response == "":                                                                                                  
            break                                                                                                                                                  
        object_names = [name.strip() for name in objects_response.split(",") if name.strip()]                                                                      
        if not object_names:                                                                                                                                       
            break                                                                                                                                                  
        # Pick up the first object in the list.                                                                                                                    
        target_object = object_names[0]                                                                                                                            
        masks, keypoints = generate_keypoints(image)                                                                                                               
        pixel_point = select_keypoint(image, masks, keypoints, f"pick up {target_object}")                                                                         
        # Ensure pixel_point is a tuple of ints for the projection.                                                                                                
        pixel_point = (int(pixel_point[0]), int(pixel_point[1]))                                                                                                   
        object_3d_point = project_to_3d(pixel_point, depth_image)                                                                                                  
        # Approach point: 0.1 m above object, with gripper pointed downwards (pitch +90 degrees).                                                                  
        set_gripper_pose(bot, object_3d_point[0], object_3d_point[1], object_3d_point[2] + 0.1, 0, 1.5708, 0)                                                      
        # Lower the gripper to 0.02 m below the object's surface.                                                                                                  
        set_gripper_pose(bot, object_3d_point[0], object_3d_point[1], object_3d_point[2] - 0.02, 0, 1.5708, 0)                                                     
        close_gripper(bot)                                                                                                                                         
        go_home(bot)                                                                                                                                               
        open_gripper(bot)                                                                                                                                          
        sleep(2.0)
```