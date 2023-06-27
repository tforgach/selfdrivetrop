import Steer
from geometry_msgs.msg import Twist


vel_msg = Twist()
deepsteer = Steer()

def drive_by_dl(cv_image):
  global speed, drive

  vel_msg.linear.x = speed
  #mid = int(cols / 2)+75  
  if drive == True:
    vel_msg.angular.z = deepsteer.predict()
    vel_msg.linear.x = speed
    velocity_pub.publish(vel_msg)
  else:
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    velocity_pub.publish(vel_msg)
  return

if __name__ == '__main__':
  rospy.init_node('deepsteer_node', anonymous=True)
  imgtopic = rospy.get_param("~imgtopic_name") # private name
  rospy.Subscriber(imgtopic, Image, image_callback)
  enable_pub = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
  velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
  try:
    drive_by_dl(cv_image)
    rospy.spin()
  except rospy.ROSInterruptException:
    pass