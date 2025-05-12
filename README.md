# Robot Interact

Launch arm


```
roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s
```

Launch camera and transform -


```
roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s use_pointcloud_tuner_gui:=true use_armtag_tuner_gui:=true
```
