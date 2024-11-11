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

def computeTriangleArea(A, B, C):
    return 0.5 * abs(np.cross(B - A, C - A))

class Straight_Line_Negative(Base_Course):

    def __init__(self, step: float, param_dict):
        super().__init__(step, param_dict)
        
        self.target_vel_on_line = 0.0
        self.target_acc_on_line = 0.0
        self.vel_idx, self.acc_idx = 0, 0

        self.on_line_vel_flag = False


    def get_trajectory_points(self, long_side_length: float, short_side_length: float, ego_point = np.array([0.0,0.0]), goal_point  = np.array([0.0,0.0])):

        total_distance = long_side_length
        t_array = np.arange(start=0.0, stop=total_distance, step=self.step).astype("float")

        self.yaw = np.pi * np.ones(len(t_array))
        self.parts = ["linear" for _ in range(len(t_array.copy()))]
        x = np.linspace(total_distance / 2, -total_distance / 2, len(t_array))
        y = np.zeros(len(t_array))

        self.points = np.vstack((x, y)).T
        self.curvature = 1e-9 * np.ones(len(t_array))
        self.achievement_rates = np.linspace(0.0, 1.0, len(t_array))
        
        return self.points, self.yaw, self.curvature, self.parts, self.achievement_rates


    def get_target_velocity(self, nearestIndex, current_vel, current_acc, collected_data_counts_of_vel_acc):

        part = self.parts[
            nearestIndex
        ]
        achievement_rate = self.achievement_rates[nearestIndex]

        acc_kp_of_pure_pursuit = self.params.acc_kp

        N_V = self.params.num_bins_v
        N_A = self.params.num_bins_a
    

        max_lateral_accel = self.params.max_lateral_accel
        max_vel_from_lateral_acc = np.sqrt(
                max_lateral_accel * self.curvature[nearestIndex]
            )
        
        min_data_num_margin = 5
        min_index_list = []
        if part == "linear" and achievement_rate < 0.05:
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


        if self.target_vel_on_line > self.params.v_max * 3.0 / 4.0:
            self.deceleration_rate = 0.55 + 0.10
        elif self.target_vel_on_line > self.params.v_max / 2.0:
            self.deceleration_rate = 0.65 + 0.10
        else:
            self.deceleration_rate = 0.85 + 0.10


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
                
        # deceleration
        if self.deceleration_rate <= achievement_rate:
            target_vel = 0.0

        return target_vel

    def get_boundary_points(self):
        return  np.vstack((self.A, self.B, self.C, self.D))
    
    def check_in_boundary(self, current_position):
        P = current_position[0:2]

        area_ABCD = computeTriangleArea(self.A, self.B, self.C) + computeTriangleArea(self.C, self.D, self.A)

        area_PAB = computeTriangleArea(P, self.A, self.B)
        area_PBC = computeTriangleArea(P, self.B, self.C)
        area_PCD = computeTriangleArea(P, self.C, self.D)
        area_PDA = computeTriangleArea(P, self.D, self.A)

        if area_PAB + area_PBC + area_PCD + area_PDA > area_ABCD * 1.001:
            return False
        else:
            return True
        