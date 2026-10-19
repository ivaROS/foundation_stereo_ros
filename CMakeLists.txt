cmake_minimum_required(VERSION 3.8)
project(foundation_stereo_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(depth_validation
  PACKAGE_DIR scripts/depth_validation)

install(PROGRAMS
  scripts/foundation_stereo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME foundation_stereo_node)

install(PROGRAMS scripts/depth_validation/analyze_bag.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME analyze_bag)
install(PROGRAMS scripts/depth_validation/analyze_sequential.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME analyze_sequential)
install(PROGRAMS scripts/depth_validation/benchmark_foundation_cpu.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME benchmark_foundation_cpu)
install(PROGRAMS scripts/depth_validation/benchmark_foundation_prepost.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME benchmark_foundation_prepost)
install(PROGRAMS scripts/depth_validation/compare_foundation_onnx.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME compare_foundation_onnx)
install(PROGRAMS scripts/depth_validation/monitor_process.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME monitor_process)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME})

ament_package()
