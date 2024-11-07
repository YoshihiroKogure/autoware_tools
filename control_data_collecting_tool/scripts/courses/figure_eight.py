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

import numpy as np
from courses.base_course import Base_Course

class Figure_Eight(Base_Course):

    def __init__(self, step: float, param_dict):
        super().__init__(step, param_dict)
        
        self.target_vel_on_line = 0.0
        self.target_acc_on_line = 0.0
        self.vel_idx, self.acc_idx = 0, 0

        self.on_line_vel_flag = False
        self.prev_part = "left_circle"
        self.deceleration_rate = 0.70

    def get_trajectory_points(self, long_side_length: float, short_side_length: float):

        a = short_side_length
        b = long_side_length

        C = [-(b / 2 - (1.0 - np.sqrt(3) / 2) * a), -a / 2]
        D = [(b / 2 - (1.0 - np.sqrt(3) / 2) * a), -a / 2]

        R = a  # radius of the circle
        OL = [-b / 2 + a, 0]  # center of the left circle
        OR = [b / 2 - a, 0]  # center of the right circle
        OB = np.sqrt(
            (b / 2 + (1.0 - np.sqrt(3) / 2) * a) ** 2 + (a / 2) ** 2
        )  # half length of the linear trajectory
        AD = 2 * OB
        θB = np.arctan(
            a / 2 / (b / 2 + (1.0 - np.sqrt(3) / 2) * a)
        )  # Angle that OB makes with respect to x-axis
        BD = 2 * np.pi * R / 6  # the length of arc BD
        AC = BD
        CO = OB

        total_distance = 4 * OB + 2 * BD

        t_array = np.arange(start=0.0, stop=total_distance, step=self.step).astype("float")
        x = np.array([0.0 for i in range(len(t_array.copy()))])
        y = np.array([0.0 for i in range(len(t_array.copy()))])
        self.yaw = t_array.copy()
        self.curvature = t_array.copy()
        self.achievement_rates = t_array.copy()
        self.parts = ["part" for _ in range(len(t_array.copy()))]
        i_end = t_array.shape[0]

        for i, t in enumerate(t_array):
            if t > OB + BD + AD + AC + CO:
                i_end = i
                break

            if 0 <= t and t <= OB:
                x[i] = (b / 2 - (1.0 - np.sqrt(3) / 2) * a) * t / OB
                y[i] = a * t / (2 * OB)
                self.parts[i] = "linear_positive"
                self.achievement_rates[i] = t / (2 * OB) + 0.5

            if OB <= t and t <= OB + BD:
                t1 = t - OB
                t1_rad = t1 / R
                x[i] = OR[0] + R * np.cos(np.pi / 6 - t1_rad)
                y[i] = OR[1] + R * np.sin(np.pi / 6 - t1_rad)
                self.parts[i] = "right_circle"
                self.achievement_rates[i] = t1 / BD

            if OB + BD <= t and t <= OB + BD + AD:
                t2 = t - (OB + BD)
                x[i] = D[0] - (b / 2 - (1.0 - np.sqrt(3) / 2) * a) * t2 / OB
                y[i] = D[1] + a * t2 / (2 * OB)
                self.parts[i] = "linear_negative"
                self.achievement_rates[i] = t2 / (2 * OB)

            if OB + BD + AD <= t and t <= OB + BD + AD + AC:
                t3 = t - (OB + BD + AD)
                t3_rad = t3 / R
                x[i] = OL[0] - R * np.cos(-np.pi / 6 + t3_rad)
                y[i] = OL[1] - R * np.sin(-np.pi / 6 + t3_rad)
                self.parts[i] = "left_circle"
                self.achievement_rates[i] = t3 / BD

            if OB + BD + AD + AC <= t and t <= OB + BD + AD + AC + CO:
                t4 = t - (OB + BD + AD + AC)
                x[i] = C[0] + (b / 2 - (1.0 - np.sqrt(3) / 2) * a) * t4 / OB
                y[i] = C[1] + a * t4 / (2 * OB)
                self.parts[i] = "linear_positive"
                self.achievement_rates[i] = t4 / (2 * OB)

        # drop rest
        x = x[:i_end]
        y = y[:i_end]
        window_size = 500
        x = np.concatenate([x[-window_size//2:], x, x[:window_size//2]])
        y = np.concatenate([y[-window_size//2:], y, y[:window_size//2]])

        x_smoothed = np.convolve(x, np.ones(window_size)/window_size, mode='valid')[window_size//2:-window_size//2]
        y_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='valid')[window_size//2:-window_size//2]
        self.trajectory_points = np.array([x_smoothed, y_smoothed]).T
        self.parts = self.parts[:i_end]
        self.achievement_rates = self.achievement_rates[:i_end]


        dx = (x_smoothed[1:] - x_smoothed[:-1]) / self.step
        dy = (y_smoothed[1:] - y_smoothed[:-1]) / self.step

        ddx = (dx[1:] - dx[:-1]) / self.step
        ddy = (dy[1:] - dy[:-1]) / self.step

        self.yaw = np.arctan2(dy, dx)
        self.yaw = np.array(self.yaw.tolist() + [self.yaw[-1]])

        self.curvature = 1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
        self.curvature = np.array(self.curvature.tolist() + [self.curvature[-2], self.curvature[-1]])

        return self.trajectory_points, self.yaw, self.curvature, self.parts, self.achievement_rates

    def get_target_velocity(self, nearestIndex, current_vel, current_acc, collected_data_counts_of_vel_acc):

        part = self.parts[
            nearestIndex
        ]
        self.prev_part = part
        achievement_rate = self.achievement_rates[nearestIndex]

        acc_kp_of_pure_pursuit = self.params.acc_kp

        N_V = self.params.num_bins_v
        N_A = self.params.num_bins_a
    

        max_lateral_accel = self.params.max_lateral_accel
        max_vel_from_lateral_acc = np.sqrt(
                max_lateral_accel / self.curvature[nearestIndex]
            )
        
        target_vel = np.min([max_vel_from_lateral_acc, 6.0])

        min_data_num_margin = 5
        min_index_list = []
        if (self.prev_part == "left_circle" or self.prev_part == "right_circle") and (part == "linear_positive" or part == "linear_negative"):
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

            for i in range(self.params.collecting_data_min_n_v, self.params.collecting_data_max_n_v):
                for j in range(self.params.collecting_data_min_n_a, self.params.collecting_data_max_n_a):
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

        if part == "linear_positive" or part == "linear_negative":
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
                        [self.target_vel_on_line - 1.5 * self.params.v_max / N_V, self.params.v_max / N_V / 2.0]
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
                target_vel = np.max([target_vel, 0.5])
                    
            # deceleration
            if self.deceleration_rate <= achievement_rate:
                target_vel = np.sqrt(  max_lateral_accel / max(self.curvature) )

        # set target velocity on circle part
        if part == "left_circle" or part == "right_circle":
            if achievement_rate < 0.10 and self.target_vel_on_line > self.params.v_max / 2.0:
                target_vel = np.sqrt(  max_lateral_accel / max(self.curvature) )
            elif achievement_rate < 0.50:
                target_vel = max_vel_from_lateral_acc / 2.0
            else:
                target_vel = max_vel_from_lateral_acc

        return target_vel
