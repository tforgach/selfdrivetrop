cmake_minimum_required(VERSION 3.0.2)
project(simple_sim_h_working)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)
generate_dynamic_reconfigure_options(cfg/Follow.cfg)

catkin_package( )

include_directories(
  ${catkin_INCLUDE_DIRS}
)
