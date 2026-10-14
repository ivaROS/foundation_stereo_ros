cmake_minimum_required(VERSION 3.8)
project(foundation_stereo_ros)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/foundation_stereo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME foundation_stereo_node)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

ament_package()
