#!/usr/bin/env python3
# Stop at Crosswalk

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from simple_sim_h_working.cfg import FollowConfig   # packageName.cfg
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import numpy as np
empty_msg = Empty()
vel_msg = Twist()
bridge = CvBridge()

def dyn_rcfg_cb(config, level):
  global thresh, drive,speed, contour_max, height, medBlur, ellipse, gray,turnMult, middle
  thresh = config.thresh
  drive = config.enable_follow
  speed = config.speed
  contour_max = config.contour
  height = config.height
  medBlur = config.median_blur
  ellipse = config.ellipse
  gray = config.gray
  turnMult = config.turn
  middle = config.middle
  #thresh=253
  #speed = 0.5
  #drive = True
  return config

def stop_drive():
   print("Stopped")
   vel_msg.linear.x = 0
   enable_pub.publish(empty_msg)
   velocity_pub.publish(vel_msg)
   rate.sleep()
   
def image_callback(ros_image):
  global thresh, drive, speed, contour_max, medBlur,rate, ellipse,gray, turnMult, middle
  rate = rospy.Rate(20) # Default rate  

  #thresh = 100
  #drive = True
  global bridge
  try: #convert ros_image into an opencv-compatible imageadi
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
  except CvBridgeError as e:
    print(e)

  #Running the car
  vel_msg.linear.x = speed
  enable_pub.publish(empty_msg)
  velocity_pub.publish(vel_msg) # complete the block <=======   
  
  
  cv_image_org = cv_image
  
  
  # from now on, you can work exactly like with opencv
  cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
  print(cv_image.shape)
  (rows,cols) = cv_image.shape#,channels
 
  
  new_rows = int(rows*(1-height))
  cv_image = cv_image[new_rows:rows, 00:cols-0]#, 0:channels 150
  img_hist = cv2.equalizeHist(cv_image)
  #cv2.imshow('hist',img_hist)
  median = cv2.medianBlur(img_hist,medBlur)
  #cv2.imshow('median',median) 

  
  #rows=int(rows*1/3)
   #   else:
  gray_image = median #cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
  #hsv = cv2.cvtColor(cv_image_org, cv2.COLOR_BGR2HSV)
  
  #print(thresh)
  ret, thresh_img = cv2.threshold(gray_image, # input image
                                thresh,     # threshol_value,
                                255,        # max value in image
                                cv2.THRESH_BINARY) # threshold type

  blur = cv2.blur(gray_image, (gray,gray)) #gray_image
  #ret, thresh_img = cv2.threshold(blur, thresh, 225, cv2.THRESH_BINARY)
  #thresh_img = []

 # sensitivity = 255 - thresh
 # lower_white = np.array([0,0,255-sensitivity])
 # upper_white = np.array([255,sensitivity,255]) #Why did I make these like this?

 # thresh_img = cv2.inRange(blur,lower_white, upper_white)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ellipse,ellipse))
  dilated = cv2.dilate(thresh_img,kernel)
 # kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE,(9,9))#Comment out if not working
 # new_contour = cv2.morphologyEx(dilated.copy(),kernel)#This too
  
  contours, heirarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #dilated.copy()

  #num_white_pix = cv2.countNonZero(bw_image)
  #white_pct = (100* num_white_pix) / (rows * cols)
  #font = cv2.FONT_HERSHEY_SIMPLEX
  #for c in contours:
    #print(cv2.contourArea(c))
  areas = [cv2.contourArea(c) for c in contours]
  #areas = [cv2.contourArea(c) for c in contours]
  #print("1:",areas)
  #while(max(areas)>contour_max):
  #  areas.remove(max(areas))
  #print("max",max(areas))
  
  max_index = np.argmax(areas)
  cnt=contours[max_index]    
  #contours2 = np.copy(contours)
  #contours_original = contours
  #contours2.delete(cnt, np.where(contours2 == cnt))
  #cntList2 = contours2.tolist()
  #cntList2.remove(cnt)
  #areas = [cv2.contourArea(c) for c in contours]
  #print("1:",areas)
  #while(max(areas)>20000):
  #  areas.remove(max(areas))
  #print("max",max(areas))
  #areas.remove(max(areas))
  #areas[areas.index(max(areas))] = 0 
  #print("2:",np.argmax(areas))
  #areas = [cv2.contourArea(c) for c in contours2]
  #cnt2 = contours[np.argmax(areas)]
  #cx2 = 0
  M = cv2.moments(cnt)
  if M['m00'] != 0:
    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
  #print(areas)
  cv2.drawContours(cv_image, contours, -1, (0,0,255), 10)
  #if(areas[np.argmax(areas)] >600):
  #  M2 = cv2.moments(cnt2)
  #  if M2['m00'] != 0:
  #    cx2,cy2 = int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00'])150
  #    cv2.circle(cv_image, (cx2, cy2), 7, (0, 0, 255), -1)
        #print(f"x = {cx}     y = {cy}")

  cv2.circle(cv_image, (cx, cy), 7, (0, 0, 255), -1)
  
  cv2.imshow('Contours', cv_image)
  #(rows,cols) = cv_image.shape
  #print(cx)
  if drive: # and contours
    vel_msg.linear.x = speed 
     #224
    if(abs(cx-middle)>50):
   #   if abs(cx-cx2)>14:#abs(cx-middle)>14
      vel_msg.angular.z = turnMult*-0.6*(cx-middle)/middle
      if(abs(cx-middle)>180):
        vel_msg.angular.z = turnMult*-1.2*(cx-middle)/middle
      elif(abs(cx-middle)>300):
        vel_msg.angular.z = turnMult*-1.5*(cx-middle)/middle
   #     vel_msg.angular.z = -2.0*((abs(cx-middle)/middle)-(abs(cx2-middle)/middle))
   #   else:
   #     vel_msg.angular.z = 0
   # elif abs(cx-middle)>140:
   #   vel_msg.angular.z = -1.0*(cx-middle)/middle
    else:
      vel_msg.angular.z = 0
      
  #  if(cx<middle-14):
  #    vel_msg.angular.z = -(cx-middle)/middle
  #    
  #  elif cx>middle+14:
  #    vel_msg.angular.z = -(cx-middle)/middle
  #  else:
  #    vel_msg.angular.z = 0
    print("Driving at ",vel_msg)
    enable_pub.publish(empty_msg)
    velocity_pub.publish(vel_msg) # complete the block <=======   
    rate.sleep()
      
  else:
     stop_drive()# drive disabled. Complete the following block. <======
    
    

  #cv2.imshow("My Image Window", bw_image)
  rate.sleep()
  cv2.waitKey(1)
  
if __name__ == '__main__':
  rospy.init_node('follow_lane', anonymous=True)
  imgtopic = rospy.get_param("~imgtopic_name") # private name
  rospy.Subscriber(imgtopic, Image, image_callback)
  enable_pub = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
  velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
  srv = Server(FollowConfig, dyn_rcfg_cb)
  try:
    rospy.spin()
  except rospy.ROSInterruptException:
    pass
