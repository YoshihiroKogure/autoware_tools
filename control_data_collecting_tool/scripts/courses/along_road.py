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
import matplotlib.pyplot as plt

def resample_curve(x, y, step_size):
    # Calculate the distance between each point and find the cumulative distance
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
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

    def get_trajectory_points(
        self, long_side_length: float, short_side_length: float, ego_point, goal_point
    ):
        x, y = self.handler.get_shortest_path(ego_point, goal_point)
        x, y = resample_curve(x, y, self.step)

        if x is None or y is None:
            return None

        self.trajectory_points = np.array([x, y]).T
        self.parts = []
        self.achievement_rates = np.zeros(len(x))


        dx = (x[1:] - x[:-1]) / self.step
        dy = (y[1:] - y[:-1]) / self.step
        self.yaw = np.arctan2(dy, dx)
        self.yaw = np.array(self.yaw.tolist() + [self.yaw[-1]])


        window_size = 100
        x = np.concatenate((x[0] * np.ones(window_size // 2), x, x[-1] * np.ones(window_size // 2 - 1)))
        y = np.concatenate((y[0] * np.ones(window_size // 2), y, y[-1] * np.ones(window_size // 2 - 1)))

        x_smoothed = np.convolve(x, np.ones(window_size) / window_size, mode="valid")
        y_smoothed = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

        dx = (x_smoothed[1:] - x_smoothed[:-1]) / self.step
        dy = (y_smoothed[1:] - y_smoothed[:-1]) / self.step

        ddx = (dx[1:] - dx[:-1]) / self.step
        ddy = (dy[1:] - dy[:-1]) / self.step

        self.curvature = (
            1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
        )
        self.curvature = np.array(
            self.curvature.tolist() + [self.curvature[-1], self.curvature[-1]]
        )

        for i in range(len(self.curvature)):
            if self.curvature[i] < 10**(-2):
                self.parts.append("linear")
            else:
                self.parts.append("curve")

        previous_part = "curve"
        start_index = []
        end_index = []
        for i, part in enumerate(self.parts):
            current_part = part

            if previous_part == "curve" and current_part == "linear":
                start_index.append(i)

            if previous_part == "linear" and current_part == "curve":
                end_index.append(i-1)

            if i == len(self.parts) - 1 and len(start_index) > len(end_index):
                end_index.append(i-1)
            
            previous_part = current_part

        for i in range(len(start_index)):
            st = start_index[i]
            ed = end_index[i]

            if (ed -st) * self.step > 50.0:
                self.achievement_rates[st:ed+1] = np.linspace(0.0, 1.0, ed-st+1)

        for i in range(len(self.parts)):
            if self.achievement_rates[i] > 0.0:
                self.parts[i] = "linear"
            else:
                self.parts[i] = "curve"

        plt.plot(self.achievement_rates)
        plt.show()


    def get_target_velocity(
        self, nearestIndex, current_time, current_vel, current_acc, collected_data_counts_of_vel_acc
    ):
        
        part = self.parts[nearestIndex]
        achievement_rate = self.achievement_rates[nearestIndex]

        acc_kp_of_pure_pursuit = self.params.acc_kp

        N_V = self.params.num_bins_v
        N_A = self.params.num_bins_a

        max_lateral_accel = self.params.max_lateral_accel
        max_vel_from_lateral_acc = np.sqrt(max_lateral_accel * self.curvature[nearestIndex])

        min_data_num_margin = 5
        min_index_list = []
        if (part == "linear" and self.previous_part == "curve") or (part == "linear" and achievement_rate < 0.05):
            self.on_line_vel_flag = True
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
        self.previous_part = part

        
        self.deceleration_rate = 0.55
        if (
            current_vel > self.target_vel_on_line - self.params.v_max / N_V / 8.0
            and self.target_vel_on_line >= self.params.v_max / 2.0
        ):
            self.on_line_vel_flag = False

        elif (
            abs(current_vel - self.target_vel_on_line) < self.params.v_max / N_V / 4.0
            and self.target_vel_on_line < self.params.v_max / 2.0
        ):
            self.on_line_vel_flag = False

        # accelerate until vehicle reaches target_vel_on_line
        if 0.0 <= achievement_rate and achievement_rate < 0.45 and self.on_line_vel_flag:
            target_vel = self.target_vel_on_line

            if (
                current_vel > self.target_vel_on_line - self.params.v_max / N_V * 0.5
                and self.target_acc_on_line > 2.0 * self.params.a_max / N_A
            ):
                target_vel = current_vel + self.target_acc_on_line / acc_kp_of_pure_pursuit

        # collect target_acceleration data when current velocity is close to target_vel_on_line
        elif (
            achievement_rate < self.deceleration_rate
            or self.target_vel_on_line < self.params.v_max / 2.0
        ):
            if collected_data_counts_of_vel_acc[self.vel_idx, self.acc_idx] > 50:
                self.acc_idx = np.argmin(collected_data_counts_of_vel_acc[self.vel_idx, :])
                self.target_acc_on_line = self.params.a_bin_centers[self.acc_idx]

            if (
                current_vel
                < max(
                    [
                        self.target_vel_on_line - 1.5 * self.params.v_max / N_V,
                        self.params.v_max / N_V / 2.0,
                    ]
                )
                and self.target_acc_on_line < 0.0
            ):
                self.acc_idx = np.argmin(
                    collected_data_counts_of_vel_acc[self.vel_idx, int(N_A / 2.0) : N_A]
                ) + int(N_A / 2)
                self.target_acc_on_line = self.params.a_bin_centers[self.acc_idx]

            elif (
                current_vel > self.target_vel_on_line + 1.5 * self.params.v_max / N_V
                and self.target_acc_on_line > 0.0
            ):
                self.acc_idx = np.argmin(
                    collected_data_counts_of_vel_acc[self.vel_idx, 0 : int(N_A / 2.0)]
                )
                self.target_acc_on_line = self.params.a_bin_centers[self.acc_idx]

            target_vel = current_vel + self.target_acc_on_line / acc_kp_of_pure_pursuit

        # deceleration
        if self.deceleration_rate <= achievement_rate:
            target_vel = (5.0 * (achievement_rate - self.deceleration_rate) + 3.5 * (1 - achievement_rate) ) / (1.0 - self.deceleration_rate)
        
        if part == "curve":
            target_vel = 3.5

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
