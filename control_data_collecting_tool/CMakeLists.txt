cmake_minimum_required(VERSION 3.16)
project(control_data_collecting_tool)

# Autoware CMake package configuration
find_package(autoware_cmake REQUIRED)
autoware_package()

# Required ROS 2 packages
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(autoware_map_msgs REQUIRED)

# Add Lanelet2 configuration
find_package(lanelet2_core REQUIRED)
find_package(lanelet2_traffic_rules REQUIRED)
find_package(lanelet2_routing REQUIRED)
find_package(autoware_lanelet2_extension REQUIRED)

# Qt5 configuration
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
set(QT_LIBRARIES Qt5::Widgets)
set(CMAKE_AUTOMOC ON)

# Python and pybind11 configuration
set(Python3_EXECUTABLE "/usr/bin/python3")
find_package(pybind11 REQUIRED)

# Set include paths for Python headers
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${autoware_lanelet2_extension_INCLUDE_DIRS})

# Define libraries for Autoware
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/data_collecting_area_selection.cpp
  src/data_collecting_goal_pose.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  Qt5
  rviz_common
  pluginlib
  rclcpp
  autoware_map_msgs
)

# Configure the Pybind11 module manually
add_library(lanelet_pybind_module SHARED
  src/lanelet_pybind_wrapper.cpp
)

# Link the necessary targets
target_link_libraries(lanelet_pybind_module PRIVATE
  ${autoware_map_msgs_LIBRARIES}
  ${autoware_lanelet2_extension_LIBRARIES}
  lanelet2_core
  lanelet2_traffic_rules
  lanelet2_routing
  pybind11::module
  ${QT_LIBRARIES}
)

# Set module properties
set_target_properties(lanelet_pybind_module PROPERTIES
  OUTPUT_NAME "lanelet_pybind_module"
  PREFIX ""
)

# Add necessary include directories
include_directories(
  ${autoware_map_msgs_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Export the plugin XML file
pluginlib_export_plugin_description_file(rviz_common plugins/plugin_description.xml)

# Install Python scripts
install(PROGRAMS
  scripts/data_collecting_pure_pursuit_trajectory_follower.py
  scripts/data_collecting_trajectory_publisher.py
  scripts/data_collecting_plotter.py
  scripts/data_collecting_rosbag_record.py
  scripts/data_collecting_data_counter.py
  scripts/data_collecting_base_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install other configurations and files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Install Pybind11 module
install(TARGETS lanelet_pybind_module
  DESTINATION lib/${PROJECT_NAME}
)

# Package configuration
ament_auto_package(
  INSTALL_TO_SHARE
  plugins
)
