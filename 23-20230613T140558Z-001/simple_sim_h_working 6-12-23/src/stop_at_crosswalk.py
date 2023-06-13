#!/usr/bin/env python3
# Stop at Crosswalk

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from stop_at_cw_pkg.cfg import StopAtCwConfig   # packageName.cfg
from geometry_msgs.msg import Twist

vel_msg = Twist()
bridge = CvBridge()

def dyn_rcfg_cb(config, level):
  global thresh, drive
  thresh = config.thresh
  drive = config.enable_drive
  return config

def stop_drive():
   vel_msg.linear.x = 0
   velocity_pub.publish(vel_msg)
def image_callback(ros_image):
  global bridge
  try: #convert ros_image into an opencv-compatible imageadi
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
  except CvBridgeError as e:
    print(e)
  
  # from now on, you can work exactly like with opencv
  cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
  (rows,cols,channels) = cv_image.shape

  gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
  ret, bw_image = cv2.threshold(gray_image, # input image
                                thresh,     # threshol_value,
                                255,        # max value in image
                                cv2.THRESH_BINARY) # threshold type

  num_white_pix = cv2.countNonZero(bw_image)
  white_pct = (100* num_white_pix) / (rows * cols)
  font = cv2.FONT_HERSHEY_SIMPLEX

  if drive == True:
    if white_pct > 30:
      cv2.putText(bw_image,f"Auto STOP = {white_pct:.1f}%",(10,rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
      stop_drive()
    else:
       vel_msg.linear.x = 1 
       velocity_pub.publish(vel_msg) # complete the block <=======
      
      
      
  else:
     stop_drive()# drive disabled. Complete the following block. <======
    
    

  cv2.imshow("My Image Window", bw_image)
  cv2.waitKey(3)
  
if __name__ == '__main__':
  rospy.init_node('stop_at_crosswalk', anonymous=True)
  imgtopic = rospy.get_param("~imgtopic_name") # private name
  rospy.Subscriber(imgtopic, Image, image_callback)
  velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
  srv = Server(StopAtCwConfig, dyn_rcfg_cb)
  try:
    rospy.spin()
  except rospy.ROSInterruptException:
    pass
