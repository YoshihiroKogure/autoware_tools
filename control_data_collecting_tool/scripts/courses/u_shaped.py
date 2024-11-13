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
import numpy as np


class U_Shaped(Base_Course):
    def __init__(self, step: float, param_dict):
        super().__init__(step, param_dict)

        self.target_vel_on_line = 0.0
        self.target_acc_on_line = 0.0
        self.vel_idx, self.acc_idx = 0, 0

        self.on_line_vel_flag = False
        self.prev_part = "left_circle"
        self.deceleration_rate = 0.70

    def get_trajectory_points(
        self,
        long_side_length: float,
        short_side_length: float,
        ego_point=np.array([0.0, 0.0]),
        goal_point=np.array([0.0, 0.0]),
    ):
        a = short_side_length
        b = long_side_length

        # Boundary points between circular and linear trajectory
        A = [-(b - a) / 2, a / 2]
        B = [(b - a) / 2, a / 2]
        C = [-(b - a) / 2, -a / 2]
        D = [(b - a) / 2, -a / 2]

        R = a / 2  # radius of the circle
        OL = [-(b - a) / 2, 0]  # center of the left circle
        OR = [(b - a) / 2, 0]  # center of the right circle

        AB = b - a
        arc_BD = np.pi * R
        DC = b - a
        arc_CA = np.pi * R

        total_distance = 2 * AB + 2 * np.pi * R

        t_array = np.arange(start=0.0, stop=total_distance, step=self.step).astype("float")
        x = np.array([0.0 for i in range(len(t_array.copy()))])
        y = np.array([0.0 for i in range(len(t_array.copy()))])
        self.achievement_rates = t_array.copy()
        self.parts = ["part" for _ in range(len(t_array.copy()))]
        i_end = t_array.shape[0]

        for i, t in enumerate(t_array):
            if t > AB + arc_BD + DC + arc_CA:
                i_end = i
                break

            if 0 <= t and t <= AB:
                section_rate = t / AB
                x[i] = section_rate * B[0] + (1 - section_rate) * A[0]
                y[i] = section_rate * B[1] + (1 - section_rate) * A[1]
                self.parts[i] = "linear_positive"
                self.achievement_rates[i] = section_rate

            if AB <= t and t <= AB + arc_BD:
                section_rate = (t - AB) / arc_BD
                x[i] = OR[0] + R * np.cos(np.pi / 2 - np.pi * section_rate)
                y[i] = OR[1] + R * np.sin(np.pi / 2 - np.pi * section_rate)
                self.parts[i] = "right_circle"
                self.achievement_rates[i] = section_rate

            if AB + arc_BD <= t and t <= AB + arc_BD + DC:
                section_rate = (t - AB - arc_BD) / DC
                x[i] = section_rate * C[0] + (1 - section_rate) * D[0]
                y[i] = section_rate * C[1] + (1 - section_rate) * D[1]
                self.parts[i] = "linear_negative"
                self.achievement_rates[i] = section_rate

            if AB + arc_BD + DC <= t and t <= AB + arc_BD + DC + arc_CA:
                section_rate = (t - AB - arc_BD - DC) / arc_CA
                x[i] = OL[0] + R * np.cos(3 * np.pi / 2 - np.pi * section_rate)
                y[i] = OL[1] + R * np.sin(3 * np.pi / 2 - np.pi * section_rate)
                self.parts[i] = "left_circle"
                self.achievement_rates[i] = section_rate

        # drop rest
        x = x[:i_end]
        y = y[:i_end]
        self.trajectory_points = np.array([x, y]).T

        dx = (x[1:] - x[:-1]) / self.step
        dy = (y[1:] - y[:-1]) / self.step
        ddx = (dx[1:] - dx[:-1]) / self.step
        ddy = (dy[1:] - dy[:-1]) / self.step

        self.yaw = np.arctan2(dy, dx)
        self.yaw = np.array(self.yaw.tolist() + [self.yaw[-1]])
        self.curvature = (
            1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
        )
        self.curvature = np.array(
            self.curvature.tolist() + [self.curvature[-2], self.curvature[-1]]
        )

        self.parts = self.parts[:i_end]
        self.achievement_rates = self.achievement_rates[:i_end]

        return self.trajectory_points, self.yaw, self.curvature, self.parts, self.achievement_rates

    def get_target_velocity(
        self, nearestIndex, current_time, current_vel, current_acc, collected_data_counts_of_vel_acc
    ):
        part = self.parts[nearestIndex]
        self.prev_part = part
        achievement_rate = self.achievement_rates[nearestIndex]

        acc_kp_of_pure_pursuit = self.params.acc_kp

        N_V = self.params.num_bins_v
        N_A = self.params.num_bins_a

        max_lateral_accel = self.params.max_lateral_accel
        max_vel_from_lateral_acc = np.sqrt(max_lateral_accel / self.curvature[nearestIndex])

        target_vel = np.min([max_vel_from_lateral_acc, 6.0])

        min_data_num_margin = 5
        min_index_list = []
        if (self.prev_part == "left_circle" or self.prev_part == "right_circle") and (
            part == "linear_positive" or part == "linear_negative"
        ):
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
                target_vel = np.max([target_vel, 0.5])

            # deceleration
            if self.deceleration_rate <= achievement_rate:
                target_vel = np.sqrt(max_lateral_accel / max(self.curvature))

        # set target velocity on circle part
        if part == "left_circle" or part == "right_circle":
            if achievement_rate < 0.10 and self.target_vel_on_line > self.params.v_max / 2.0:
                target_vel = np.sqrt(max_lateral_accel / max(self.curvature))
            elif achievement_rate < 0.50:
                target_vel = max_vel_from_lateral_acc / 2.0
            else:
                target_vel = max_vel_from_lateral_acc

        return target_vel
