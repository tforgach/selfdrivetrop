import rosbag
import csv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
from collections import deque

bag_file = "/home/brendan/catkin_ws/src/deepsteer/bag4.bag" #CHANGE BASED ON YOUR COMPUTER, THIS IS THE BAG LOCATION
image_topic = "/camera/image_raw"
twist_topic = "/vehicle/twist"
image_folder = "/home/brendan/catkin_ws/src/deepsteer/images" #CHANGE BASED ON YOUR COMPUTER, THIS IS WHERE IT WILL PUT IMAGES
twist_csv_file = "/home/brendan/catkin_ws/src/deepsteer/twist.csv" #CHANGE BASED ON YOUR COMPUTER, THIS IS THE CSV FILE THAT IS CREATED/USED

# Instantiate bag, bridge and csv writer
bag = rosbag.Bag(bag_file, "r")
bridge = CvBridge()
csv_file = open(twist_csv_file, "w")
csv_writer = csv.writer(csv_file)

# Write the titles in the first line
csv_writer.writerow(['image_path', 'linear_x', 'angular_z'])

# Extract images and twist messages with time stamps
images = []
twists = []
twist_counter = 0
last_twist = None
for topic, msg, t in bag.read_messages(topics=[image_topic, twist_topic]):
    if topic == image_topic:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        timestamp = msg.header.stamp.to_sec()
        images.append((timestamp, cv_image))
    elif topic == twist_topic:
        twist_counter += 1
        twist = (msg.twist.linear.x, msg.twist.angular.z)
        timestamp = msg.header.stamp.to_sec()
        last_twist = (timestamp, twist)
        if twist_counter % 10 == 0:
            twists.append(last_twist)

# If there are more images than twists, assign the last twist to the remaining images
if len(images) > len(twists):
    for _ in range(len(images) - len(twists)):
        twists.append(last_twist)

# Sort the lists by time stamp
images.sort(key=lambda x: x[0])
twists.sort(key=lambda x: x[0])

# Write the image paths and corresponding twists to the CSV file
for i in range(len(images)):
    img_timestamp, img = images[i]
    twist_timestamp, twist = twists[i]
    img_filename = os.path.join(image_folder, str(img_timestamp) + ".png")
    cv2.imwrite(img_filename, img)
    csv_writer.writerow([img_filename, twist[0], twist[1]])

csv_file.close()

bag.close()
