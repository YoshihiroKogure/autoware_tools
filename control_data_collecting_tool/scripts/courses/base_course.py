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

from params import Params
import numpy as np

class Base_Course:

    def __init__(self, step: float, param_dict):

        self.step = step
        self.trajectory_points = None
        self.yaw = None
        self.parts = None
        self.curvature = None
        self.achievement_rates = None

        self.boundary_points = None
        self.boundary_yaw = None

        self.A = np.array([0.0, 0.0])
        self.B = np.array([0.0, 0.0])
        self.C = np.array([0.0, 0.0])
        self.D = np.array([0.0, 0.0])

        self.params = Params(param_dict)
        

    def get_trajectory_points(self, long_side_length: float, short_side_length: float, ego_point = np.array([0.0,0.0]), goal_point  = np.array([0.0,0.0])):
        pass

    def update_trajectory_points(self):
        pass

    def get_target_velocity(self, nearestIndex, current_vel, current_acc):
        pass

    def set_vertices(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def get_boundary_points(self):
        pass

    def check_in_boundary(self, current_position):
        pass

    def return_trajectory_points(self, yaw_offset, rectangle_center_position):

        rot_matrix = np.array(
            [
                [np.cos(yaw_offset), -np.sin(yaw_offset)],
                [np.sin(yaw_offset), np.cos(yaw_offset)],
            ]
        )

        trajectory_position_data = (rot_matrix @ self.trajectory_points.T).T
        trajectory_position_data += rectangle_center_position
        trajectory_yaw_data = self.yaw + yaw_offset

        return trajectory_position_data, trajectory_yaw_data, self.curvature, self.parts, self.achievement_rates
    