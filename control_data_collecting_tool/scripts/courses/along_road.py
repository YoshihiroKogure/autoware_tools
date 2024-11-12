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
from scipy.interpolate import interp1d
from courses.base_course import Base_Course

from courses.lanelet import LaneletMapHandler

def resample_curve(x, y, step_size):
    
    # Calculate the distance between each point and find the cumulative distance
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    num_samples = int(cumulative_distances[-1] / step_size)
    # Calculate new distances for resampling at equal intervals along the cumulative distance
    new_distances = np.linspace(0, cumulative_distances[-1], num_samples)
    
    # Interpolate x and y based on the distances, then resample
    x_interp = interp1d(cumulative_distances, x, kind='linear')
    y_interp = interp1d(cumulative_distances, y, kind='linear')
    new_x = x_interp(new_distances)
    new_y = y_interp(new_distances)
    
    # Return the resampled points along the curve
    return new_x, new_y

class Along_Road(Base_Course):

    def __init__(self, step: float, param_dict):
        super().__init__(step, param_dict)
        
        map_path = "/home/kogure/autoware_map/BS_map/lanelet2_map.osm"
        self.handler = LaneletMapHandler(map_path)

    def get_trajectory_points(self, long_side_length: float, short_side_length: float, ego_point, goal_point):
        
        x, y = self.handler.get_shortest_path(ego_point, goal_point)
        x, y = resample_curve(x, y, self.step)
        
        if x is None or y is None:
            return None
        
        self.trajectory_points = np.array([x, y]).T
        self.parts = ["part" for _ in range(len(x))]
        self.achievement_rates = np.linspace(0.0, 1.0,  len(x))

        dx = (x[1:] - x[:-1]) / self.step
        dy = (y[1:] - y[:-1]) / self.step

        ddx = (dx[1:] - dx[:-1]) / self.step
        ddy = (dy[1:] - dy[:-1]) / self.step

        self.yaw = np.arctan2(dy, dx)
        self.yaw = np.array(self.yaw.tolist() + [self.yaw[-1]])

        self.curvature = 1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
        self.curvature = np.array(self.curvature.tolist() + [self.curvature[-2], self.curvature[-1]])

        self.handler.plot_map()

        # return self.trajectory_points, self.yaw, self.curvature, self.parts, self.achievement_rates

    def get_target_velocity(self, nearestIndex, current_vel, current_acc, collected_data_counts_of_vel_acc):

        return 4.0
    
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

    def check_in_boundary(self, current_position):
        # should be modified later
        return True
