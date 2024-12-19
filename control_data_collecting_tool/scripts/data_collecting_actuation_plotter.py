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

from data_collecting_base_node import DataCollectingBaseNode
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
import seaborn as sns
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32MultiArray


class DataCollectingPlotter(DataCollectingBaseNode):
    def __init__(self):
        super().__init__("data_collecting_plotter")

        # callback for plot
        self.grid_update_time_interval = 5.0
        self.timer_plotter = self.create_timer(
            self.grid_update_time_interval,
            self.timer_callback_plotter,
        )

        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 24))
        plt.ion()
        self.collected_data_counts_of_vel_accel_pedal_input_subscription_ = self.create_subscription(
            Int32MultiArray,
            "/control_data_collecting_tools/collected_data_counts_of_vel_accel_pedal_input",
            self.subscribe_collected_data_counts_of_vel_accel_pedal_input,
            10,
        )
        self.collected_data_counts_of_vel_accel_pedal_input_subscription_

        self.collected_data_counts_of_vel_brake_pedal_input_subscription_ = self.create_subscription(
            Int32MultiArray,
            "/control_data_collecting_tools/collected_data_counts_of_vel_brake_pedal_input",
            self.subscribe_collected_data_counts_of_vel_brake_pedal_input,
            10,
        )
        self.collected_data_counts_of_vel_brake_pedal_input_subscription_

    def subscribe_collected_data_counts_of_vel_accel_pedal_input(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size
        self.collected_data_counts_of_vel_accel_pedal_input = np.array(msg.data).reshape((rows, cols))

    def subscribe_collected_data_counts_of_vel_brake_pedal_input(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size
        self.collected_data_counts_of_vel_brake_pedal_input = np.array(msg.data).reshape((rows, cols))

    def timer_callback_plotter(self):
        self.plot_data_collection_grid()
        plt.pause(0.1)

    def plot_data_collection_grid(self):

        # update collected acceleration and velocity grid
        for collection in self.axs[0].collections:
            if collection.colorbar is not None:
                collection.colorbar.remove()
        self.axs[0].cla()
        self.get_logger().info(str(self.accel_pedal_input_bin_centers))
        self.heatmap = sns.heatmap(
            self.collected_data_counts_of_vel_accel_pedal_input.T,
            annot=True,
            cmap="coolwarm",
            xticklabels=np.round(self.v_bin_centers, 2),
            yticklabels=np.round(self.accel_pedal_input_bin_centers , 3),
            ax=self.axs[0],
            linewidths=0.1,
            linecolor="gray",
        )

        self.axs[0].set_xlabel("Velocity bins")
        self.axs[0].set_ylabel("Accel pedal bins")

        for collection in self.axs[1].collections:
            if collection.colorbar is not None:
                collection.colorbar.remove()
        self.axs[1].cla()

        self.heatmap = sns.heatmap(
            self.collected_data_counts_of_vel_brake_pedal_input.T,
            annot=True,
            cmap="coolwarm",
            xticklabels=np.round(self.v_bin_centers, 2),
            yticklabels=np.round(self.brake_pedal_input_bin_centers , 3),
            ax=self.axs[1],
            linewidths=0.1,
            linecolor="gray",
        )

        # update collected steer and velocity grid
        self.axs[1].set_xlabel("Velocity bins")
        self.axs[1].set_ylabel("Brake pedal bins")

        self.fig.canvas.draw()


def main(args=None):
    rclpy.init(args=args)

    data_collecting_plot = DataCollectingPlotter()
    rclpy.spin(data_collecting_plot)

    data_collecting_plot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
