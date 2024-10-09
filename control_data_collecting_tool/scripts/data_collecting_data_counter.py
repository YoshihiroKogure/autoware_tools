#!/usr/bin/env python3

# Copyright 2024 Proxima Technology Inc, TIER IV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rosbag_play
import os

import rclpy
import numpy as np
from numpy import arctan2
from data_collecting_base_node import DataCollectingBaseNode


import numpy as np
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension, MultiArrayLayout

from collections import deque

def publish_Int32MultiArray(publisher_, array_data):

    msg = Int32MultiArray()

    flattened_data  = array_data.flatten().tolist()
    msg.data = flattened_data

    layout = MultiArrayLayout()

    dim1 = MultiArrayDimension()
    dim1.label = "rows"
    dim1.size = len(array_data)  
    dim1.stride = len(flattened_data)  

    dim2 = MultiArrayDimension()
    dim2.label = "cols"
    dim2.size = len(array_data[0])
    dim2.stride = len(array_data[0])

    layout.dim.append(dim1)
    layout.dim.append(dim2)
    
    msg.layout = layout

    publisher_.publish(msg)


class DataCollectingDataCounter(DataCollectingBaseNode):
    def __init__(self):   
        super().__init__('data_collecting_data_counter')

        self.vel_hist = deque([float(0.0)] * 200, maxlen=200)
        self.acc_hist = deque([float(0.0)] * 200, maxlen=200)

        self.timer_period_callback = 0.033  # 30ms
        self.timer_counter = self.create_timer(
            self.timer_period_callback,
            self.timer_callback_counter,
        )

        self.collected_data_counts_of_vel_acc_publisher_ = self.create_publisher(Int32MultiArray, '/control_data_collecting_tools/collected_data_counts_of_vel_acc', 10)
        self.collected_data_counts_of_vel_steer_publisher_ = self.create_publisher(Int32MultiArray, '/control_data_collecting_tools/collected_data_counts_of_vel_steer', 10)

        self.vel_hist_publisher_ = self.create_publisher(Float32MultiArray, '/control_data_collecting_tools/vel_hist', 10)
        self.acc_hist_publisher_ = self.create_publisher(Float32MultiArray, '/control_data_collecting_tools/acc_hist', 10)

        rosbag2_dir_list = [d for d in os.listdir("./") if os.path.isdir(os.path.join("./", d))]
        self.load_rosbag_data(rosbag2_dir_list)

    def load_rosbag_data(self, rosbag2_dir_list):

        for rosbag2_dir in rosbag2_dir_list:
            rosbag2_file = "./" + rosbag2_dir + "/" + rosbag2_dir + "_0.db3"
            db3converter = rosbag_play.db3Converter(rosbag2_file)
            
            load_acc_topic = db3converter.load_db3('/localization/acceleration')
            load_kinematic_topic = db3converter.load_db3('/localization/kinematic_state')
            if load_acc_topic and load_kinematic_topic:
                while True:
                    acceleration = db3converter.read_msg('/localization/acceleration')
                    kinematic_state = db3converter.read_msg('/localization/kinematic_state')
                    if acceleration is None or kinematic_state is None:
                        break

                    angular_z = kinematic_state.twist.twist.angular.z
                    wheel_base = self.get_parameter("wheel_base").get_parameter_value().double_value
                    steer = arctan2(
                        wheel_base * angular_z, kinematic_state.twist.twist.linear.x
                    )

                    if kinematic_state.twist.twist.linear.x > 1e-3:
                        self.count_observations(
                            kinematic_state.twist.twist.linear.x,
                            acceleration.accel.accel.linear.x,
                            steer,
                        )

    def count_observations(self, v, a, steer):
        v_bin = np.digitize(v, self.v_bins) - 1
        steer_bin = np.digitize(steer, self.steer_bins) - 1
        a_bin = np.digitize(a, self.a_bins) - 1

        if 0 <= v_bin < self.num_bins_v and 0 <= a_bin < self.num_bins_a:
            self.collected_data_counts_of_vel_acc[v_bin, a_bin] += 1

        if 0 <= v_bin < self.num_bins_v and 0 <= steer_bin < self.num_bins_steer:
            self.collected_data_counts_of_vel_steer[v_bin, steer_bin] += 1


    def timer_callback_counter(self):
        if self._present_kinematic_state is not None and self._present_acceleration is not None:
            # calculate steer
            angular_z = self._present_kinematic_state.twist.twist.angular.z
            wheel_base = self.get_parameter("wheel_base").get_parameter_value().double_value
            current_steer = arctan2(
                wheel_base * angular_z, self._present_kinematic_state.twist.twist.linear.x
            )

            current_vel = self._present_kinematic_state.twist.twist.linear.x
            current_acc = self._present_acceleration.accel.accel.linear.x

            if self._present_kinematic_state.twist.twist.linear.x > 1e-3:
               
                self.count_observations(
                    current_vel,
                    current_acc,
                    current_steer,
                )

                self.acc_hist.append(float(current_acc))
                self.vel_hist.append(float(current_vel))

        # Publish collected_data_counts_of_vel_acc
        publish_Int32MultiArray(self.collected_data_counts_of_vel_acc_publisher_, self.collected_data_counts_of_vel_acc)
        
        #Publish collected_data_counts_of_vel_steer
        publish_Int32MultiArray(self.collected_data_counts_of_vel_steer_publisher_, self.collected_data_counts_of_vel_steer)

        # Publish acc_hist
        msg = Float32MultiArray()
        msg.data = list(self.acc_hist)
        self.acc_hist_publisher_.publish(msg)

        # Publish vel_hist
        msg = Float32MultiArray()
        msg.data = list(self.vel_hist)
        self.vel_hist_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    data_collecting_data_counter = DataCollectingDataCounter()
    rclpy.spin(data_collecting_data_counter)

    data_collecting_data_counter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
