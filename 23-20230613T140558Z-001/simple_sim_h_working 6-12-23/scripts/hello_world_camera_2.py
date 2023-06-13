#!/usr/bin/env python3

#made by Mark Kocherovsky out of Dr. Joe DeRose's code
import rospy
import cv2 as cv
import numpy as np


from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ACTIVE_WINDOWS = []

# BarrelDetect class definition
class HelloCamera():
    def __init__(self):
        """Use the forward facing camera to detect a barrel (orange object)"""

        # Define the image subscriber
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
   
        # Define ROS rate
        self.rate = rospy.Rate(20)  # Vehicle rate

        # Loop and publish commands to vehicle
        while not rospy.is_shutdown():                

            # Sleep for time step
            self.rate.sleep()
            
        return

    #########################
    # Camera image callback
    #########################
    def camera_callback(self, rgb_msg):

        # Get the camera image and make a copy
        img = CvBridge().imgmsg_to_cv2(rgb_msg, "bgr8" )
        self.display_image('Camera View - Source', img)

    ####################
    # Display an image
    ####################
    def display_image(self, title_str, img):
        # Display the given image
        cv.namedWindow(title_str, cv.WINDOW_NORMAL)
        cv.imshow(title_str, img)
        cv.waitKey(3)

        # Add window to active window list
        if not ( title_str in ACTIVE_WINDOWS ):
            ACTIVE_WINDOWS.append(title_str)
            return



#################    
# Main function
#################

# Initialize the node and name it.
rospy.init_node('hello_camera_node')
print("Hello Camera Node Initialized")

# Start tester
try:
	HelloCamera()
except rospy.ROSInterruptException:
	pass



    
