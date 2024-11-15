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

from courses.base_course import Base_Course
from courses.lanelet import LaneletMapHandler
import numpy as np
from scipy.interpolate import interp1d
from collections import deque

def resample_curve(x, y, step_size):
    # Calculate the distance between each point and find the cumulative distance
    dx = x[:-1] - x[1:]
    dy = y[:-1] - y[1:]
    distances = np.sqrt( dx ** 2 + dy ** 2)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

    num_samples = int(cumulative_distances[-1] / step_size)
    # Calculate new distances for resampling at equal intervals along the cumulative distance
    new_distances = np.linspace(0, cumulative_distances[-1], num_samples)

    # Interpolate x and y based on the distances, then resample
    x_interp = interp1d(cumulative_distances, x, kind="linear")
    y_interp = interp1d(cumulative_distances, y, kind="linear")
    new_x = x_interp(new_distances)
    new_y = y_interp(new_distances)

    # Return the resampled points along the curve
    return new_x, new_y

class Along_Road(Base_Course):
    def __init__(self, step: float, param_dict):
        super().__init__(step, param_dict)
        self.closed = False
        map_path = "/home/kogure/autoware_map/BS_map/lanelet2_map.osm"
        self.handler = LaneletMapHandler(map_path)

        self.on_line_vel_flag = False
        self.target_vel_on_line = 0.0
        self.target_acc_on_line = 0.0
        self.vel_idx, self.acc_idx = 0, 0
        self.previous_part = "curve"

        self.acc_hist = deque([float(0.0)] * 20, maxlen = 20)
        self.vel_hist = deque([float(0.0)] * 20, maxlen = 20)
        self.previous_updated_time = 0.0
        self.previous_target_vel = 0.0
        self.near_target_vel = False

        self.deceleration_rate = 1.0

    def get_trajectory_points(self, long_side_length: float, short_side_length: float, ego_point, goal_point, straight_segment_threshold=50.0, curvature_threshold=1e-2 ):
        
        """
        Generates a trajectory and computes its characteristics, such as segment types (straight or curve)
        and achievement rates for straight segments.

        :param long_side_length: Length of the long side of the trajectory
        :param short_side_length: Length of the short side of the trajectory
        :param ego_point: Starting point coordinates
        :param goal_point: Target point coordinates
        :param straight_segment_threshold: Minimum length for a segment to be considered straight (default: 50.0 meters)
        :param curvature_threshold: Maximum curvature value to classify a segment as straight (default: 1e-2)
        """

        # Get the shortest path between ego_point and goal_point
        x, y = self.handler.get_shortest_path(ego_point, goal_point)
        if x is None or y is None:  # Exit if no valid path is found
            return None
        
        # Resample the trajectory to ensure uniform step intervals
        x, y = resample_curve(x, y, self.step)

        # Store the trajectory points as a 2D array of [x, y]
        self.trajectory_points = np.array([x, y]).T

        # Initialize segment classification (straight or curve) and achievement rates
        self.parts = []  
        self.achievement_rates = np.zeros(len(x))

        # Compute the yaw (heading angle) of the trajectory
        dx = (x[1:] - x[:-1]) / self.step
        dy = (y[1:] - y[:-1]) / self.step
        self.yaw = np.arctan2(dy, dx)  # Calculate the heading angles
        self.yaw = np.array(self.yaw.tolist() + [self.yaw[-1]])  # Extend to match the trajectory length
        
        ddx = (dx[1:] - dx[:-1]) / self.step
        ddy = (dy[1:] - dy[:-1]) / self.step
        self.curvature = 1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
        self.curvature = np.array(self.curvature.tolist() + [self.curvature[-1], self.curvature[-1]])

        # Prepare for trajectory smoothing
        window_size = 100  # Size of the moving average window
        # Extend the trajectory at both ends to apply smoothing
        augmented_x = np.concatenate((x[0] * np.ones(window_size // 2), x, x[-1] * np.ones(window_size // 2 - 1)))
        augmented_y = np.concatenate((y[0] * np.ones(window_size // 2), y, y[-1] * np.ones(window_size // 2 - 1)))

        # Compute smoothed trajectory using a moving average
        x_smoothed = np.convolve(augmented_x, np.ones(window_size) / window_size, mode="valid")
        y_smoothed = np.convolve(augmented_y, np.ones(window_size) / window_size, mode="valid")

        # Compute first derivatives (velocity) and second derivatives (acceleration) of the smoothed trajectory
        dx_smoothed = (x_smoothed[1:] - x_smoothed[:-1]) / self.step
        dy_smoothed = (y_smoothed[1:] - y_smoothed[:-1]) / self.step
        ddx_smoothed = (dx_smoothed[1:] - dx_smoothed[:-1]) / self.step
        ddy_smoothed = (dy_smoothed[1:] - dy_smoothed[:-1]) / self.step

        # Calculate the curvature of the smoothed trajectory
        smoothed_curvature = (
            1e-9 + abs(ddx_smoothed * dy_smoothed[:-1] - ddy_smoothed * dx_smoothed[:-1]) / 
            (dx_smoothed[:-1] ** 2 + dy_smoothed[:-1] ** 2 + 1e-9) ** 1.5
        )
        # Extend the curvature array to match the trajectory length
        smoothed_curvature = np.array(
            smoothed_curvature.tolist() + [smoothed_curvature[-1], smoothed_curvature[-1]]
        )

        # Classify each point in the trajectory as "straight" or "curve" based on curvature
        for i in range(len(smoothed_curvature)):
            if smoothed_curvature[i] < curvature_threshold:
                self.parts.append("straight")
            else:
                self.parts.append("curve")

        # Identify start and end indices of straight segments
        previous_part = "curve"
        start_index = []
        end_index = []
        for i, part in enumerate(self.parts):
            current_part = part

            # Detect the transition from "curve" to "straight"
            if previous_part == "curve" and current_part == "straight":
                start_index.append(i)

            # Detect the transition from "straight" to "curve"
            if previous_part == "straight" and current_part == "curve":
                end_index.append(i - 1)

            # Handle the case where the last segment ends as "straight"
            if i == len(self.parts) - 1 and len(start_index) > len(end_index):
                end_index.append(i - 1)
            
            previous_part = current_part

        # Assign achievement rates to sufficiently long straight segments
        for i in range(len(start_index)):
            st = start_index[i]
            ed = end_index[i]

            # Only assign achievement rates to straight segments longer than the threshold
            if (ed - st) * self.step > straight_segment_threshold:
                self.achievement_rates[st:ed + 1] = np.linspace(1e-4, 1.0, ed - st + 1)

        # Update segment classification based on achievement rates
        for i in range(len(self.parts)):
            if self.achievement_rates[i] > 0.0:
                self.parts[i] = "straight"
            else:
                self.parts[i] = "curve"

        # Plot the trajectory on the map to check which part is classified as straight or curve
        self.handler.plot_trajectory_on_map(self.trajectory_points, self.parts)


    def get_target_velocity(
        self, nearestIndex, current_time, current_vel, current_acc, collected_data_counts_of_vel_acc
    ):
        
        part = self.parts[nearestIndex]
        achievement_rate = self.achievement_rates[nearestIndex]

        acc_kp_of_pure_pursuit = self.params.acc_kp

        N_V = self.params.num_bins_v
        N_A = self.params.num_bins_a

        min_data_num_margin = 5
        min_index_list = []
    
        if (part == "straight" and self.previous_part == "curve") or (part == "straight" and achievement_rate < 0.1):
            
            min_num_data = 1e12

            # do not collect data when velocity and acceleration are low
            exclude_idx_list = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
            # do not collect data when velocity and acceleration are high
            exclude_idx_list += [
                (-1 + N_V, -1 + N_A),
                (-2 + N_V, -1 + N_A),
                (-3 + N_V, -1 + N_A),
                (-1 + N_V, -2 + N_A),
                (-2 + N_V, -2 + N_A),
                (-1 + N_V, -3 + N_A),
            ]

            for i in range(
                self.params.collecting_data_min_n_v, self.params.collecting_data_max_n_v
            ):
                for j in range(
                    self.params.collecting_data_min_n_a, self.params.collecting_data_max_n_a
                ):
                    if (i, j) not in exclude_idx_list:
                        if (
                            min_num_data - min_data_num_margin
                            > collected_data_counts_of_vel_acc[i, j]
                        ):
                            min_num_data = collected_data_counts_of_vel_acc[i, j]
                            min_index_list.clear()
                            min_index_list.append((j, i))

                        elif (
                            min_num_data + min_data_num_margin
                            > collected_data_counts_of_vel_acc[i, j]
                        ):
                            min_index_list.append((j, i))

            self.acc_idx, self.vel_idx = min_index_list[np.random.randint(0, len(min_index_list))]
            self.target_acc_on_line = self.params.a_bin_centers[self.acc_idx]
            self.target_vel_on_line = self.params.v_bin_centers[self.vel_idx]

            i = 0
            while self.parts[i + nearestIndex] == "straight":
                i += 1
            distance = i * self.step
            stop_distance = self.target_vel_on_line ** 2 / (2 * self.params.a_max)
            self.deceleration_rate = 1.0 - 20.0 / distance - stop_distance / distance
            
        self.previous_part = part


        T = 7.5
        sine =  np.sin( 2 * np.pi * current_time / T ) * np.sin( np.pi * current_time / T )

        if current_vel > self.target_vel_on_line:
            target_vel = self.target_vel_on_line - 1.0 + sine
        # acceleration
        elif current_vel < self.target_vel_on_line - 2.0 * abs(self.target_acc_on_line):
            target_vel = current_vel + self.params.a_max / acc_kp_of_pure_pursuit * ( 1.0 + 0.5 * sine )
        else:
            target_vel = current_vel + abs(self.target_acc_on_line) / acc_kp_of_pure_pursuit * ( 1.0 + 0.5 * sine )

        # deceleration
        if self.deceleration_rate - 0.05 <= achievement_rate and achievement_rate < self.deceleration_rate:
            target_vel =  current_vel -  abs(self.target_acc_on_line) / acc_kp_of_pure_pursuit * (1.0 + 0.5 * sine )
        elif self.deceleration_rate <= achievement_rate:
            target_vel = current_vel - self.params.a_max / acc_kp_of_pure_pursuit * ( 1.0 + 0.5 * sine )

            target_vel = max(target_vel, 4.0)
            

        if part == "curve" or nearestIndex > 0.85 * len(self.trajectory_points):
            target_vel = 4.0

        if nearestIndex > 0.95 * len(self.trajectory_points):
            target_vel = 0.0
        
        return target_vel
    
    def return_trajectory_points(self, yaw, translation):
        # no coordinate transformation is needed
        return self.trajectory_points, self.yaw, self.curvature, self.parts, self.achievement_rates

    def get_boundary_points(self):
        if self.trajectory_points is None or self.yaw is None:
            return None

        upper_boundary_points = []
        lower_boundary_points = []

        for point, yaw in zip(self.trajectory_points, self.yaw):
            normal = np.array([np.cos(yaw + np.pi / 2.0), np.sin(yaw + np.pi / 2.0)])
            upper_boundary_points.append(point + normal)
            lower_boundary_points.append(point - normal)

        lower_boundary_points.reverse()

        self.boundary_points = np.array(upper_boundary_points + lower_boundary_points)

