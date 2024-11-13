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

import math
import random
import numpy as np
import copy
from collections import deque

from courses.base_course import Base_Course

def safe_acos(value):
    return math.acos(max(-1, min(1, value)))


def calc_tangent_contact_points(x, y, r, a, b, c):
    """
    Calculate the contact points where the line ax + by = c is tangent to the circle
    centered at (x, y) with radius r.
    
    Parameters:
    x, y : float : Center coordinates of the circle
    r : float : Radius of the circle
    a, b, c : float : Coefficients of the line equation ax + by = c
    
    Returns:
    list of tuples : [(x_contact1, y_contact1), (x_contact2, y_contact2)] contact points or None if no tangent exists
    """
    # Check if the line has a non-zero b to solve for y = (c - ax) / b
    if b != 0:
        # Define coefficients for the quadratic equation A*x^2 + B*x + C = 0
        A = 1 + (a / b) ** 2
        B = -2 * (x + (a * (c - b * y)) / b**2)
        C = x**2 + ((c - b * y) / b) ** 2 - r**2

        # Calculate the discriminant
        abs_discriminant = abs(B**2 - 4 * A * C)


        # Calculate the two x coordinates of the tangent points
        sqrt_discriminant = math.sqrt(abs_discriminant)
        x_contact1 = (-B + sqrt_discriminant) / (2 * A)
        x_contact2 = (-B - sqrt_discriminant) / (2 * A)

        # Corresponding y coordinates for each x
        y_contact1 = (c - a * x_contact1) / b
        y_contact2 = (c - a * x_contact2) / b

    else:
        # If b == 0, we have a vertical line (ax = c), so x is constant
        x_contact1 = x_contact2 = c / a
        # Solve for y coordinates using the circle equation
        y_contact1 = y + math.sqrt(r**2 - (x_contact1 - x) ** 2)
        y_contact2 = y - math.sqrt(r**2 - (x_contact1 - x) ** 2)

    return [(x_contact1, y_contact1), (x_contact2, y_contact2)]


def calc_tangents_and_points(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1  # Calculate the distance components between circle centers
    d_sq = dx**2 + dy**2  # Square of the distance between centers
    d = math.sqrt(d_sq)  # Distance between centers

    # If centers coincide or one circle is nested within the other, no common tangents exist
    if d == 0 or d < abs(r1 - r2):
        return []

    tangents_and_points = []  # List to store tangents and their contact points

    # Calculate external tangents
    try:
        a = math.atan2(dy, dx)  # Angle between the centers
        b = safe_acos((r1 - r2) / d)  # Adjust angle based on radii difference for external tangents
        t1 = a + b
        t2 = a - b

        # Calculate tangents and contact points for each angle
        for t in [t1, t2]:
            cos_t = math.cos(t)
            sin_t = math.sin(t)

            a = cos_t  # x-coefficient of tangent line
            b = sin_t  # y-coefficient of tangent line
            c = a * x1 + b * y1 + r1  # Line constant term

            # Find contact points on each circle
            points1 = calc_tangent_contact_points(x1, y1, r1, a, b, c)
            points2 = calc_tangent_contact_points(x2, y2, r2, a, b, c)
            if points1 is not None and points2 is not None:
                tangents_and_points.append(
                    {
                        "tangent": (a, b, c),
                        "contact_point1": points1[0],
                        "contact_point2": points2[0],
                    }
                )
    except ValueError:
        pass  # Handle cases where no solution exists for external tangents

    # Calculate internal tangents
    try:
        c = safe_acos((r1 + r2) / d)  # Adjust angle based on radii sum for internal tangents
        t3 = a + c
        t4 = a - c

        # Calculate tangents and contact points for each angle
        for t in [t3, t4]:
            cos_t = math.cos(t)
            sin_t = math.sin(t)

            a = cos_t  # x-coefficient of tangent line
            b = sin_t  # y-coefficient of tangent line
            c = a * x1 + b * y1 + r1  # Line constant term

            # Find contact points on each circle
            points1 = calc_tangent_contact_points(x1, y1, r1, a, b, c)
            points2 = calc_tangent_contact_points(x2, y2, r2, a, b, c)

            if points1 is not None and points2 is not None:
                tangents_and_points.append(
                    {
                        "tangent": (a, b, c),
                        "contact_point1": points1[0],
                        "contact_point2": points2[0],
                    }
                )
    except ValueError:
        pass  # Handle cases where no solution exists for internal tangents

    return tangents_and_points  # Return list of tangents and contact points


def calc_excircle_tangent_to_two_circles(x1, y1, x2, y2, r, r_ext):
    # Distance between the centers of the two circles
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check for concentric circles, where an excircle is not possible
    if d == 0:
        raise ValueError("An excircle does not exist for concentric circles.")

    # Calculate the center of the excircle
    R_r = r_ext - r  # Difference between the excircle radius and the given circles' radius
    theta = np.arctan2(np.sqrt(R_r**2 - (d / 2) ** 2), d / 2)  # Angle relative to the line between centers

    # Coordinates of the excircle’s center
    x_center = (x2 - R_r * np.cos(theta) + x1 - R_r * np.cos(np.pi - theta)) / 2
    y_center = (y2 - R_r * np.sin(theta) + y1 - R_r * np.sin(np.pi - theta)) / 2

    # Calculate the contact point coordinates on each circle
    contact_1_x = x1 + r * np.cos(np.pi - theta)
    contact_1_y = y1 + r * np.sin(np.pi - theta)

    contact_2_x = x2 + r * np.cos(theta)
    contact_2_y = y2 + r * np.sin(theta)

    # Return angle of the tangent line, excircle center, and contact points on each circle
    return (
        np.pi - 2 * theta,  # Angle of the tangent line relative to the x-axis
        [x_center, y_center],  # Center of the excircle
        [contact_1_x, contact_1_y],  # Contact point on the first circle
        [contact_2_x, contact_2_y],  # Contact point on the second circle
    )


def calc_two_circles_one_line_trajectory(
    center_point1,
    r1,
    theta_1_start,
    theta_1_end,
    center_point2,
    r2,
    theta_2_start,
    theta_2_end,
    step,
    amplitude=0.001,
):
    # A = [
    #    center_point1[0] + r1 * np.cos(theta_1_start),
    #    center_point1[1] + r1 * np.sin(theta_1_start),
    # ]
    B = [center_point1[0] + r1 * np.cos(theta_1_end), center_point1[1] + r1 * np.sin(theta_1_end)]
    C = [
        center_point2[0] + r2 * np.cos(theta_2_start),
        center_point2[1] + r2 * np.sin(theta_2_start),
    ]

    circle_AB = r1 * abs(theta_1_end - theta_1_start)
    circle_CD = r2 * abs(theta_2_end - theta_2_start)
    BC = np.linalg.norm(np.array(B) - np.array(C))

    total_distance = circle_AB + BC + circle_CD
    t_array = np.arange(start=0.0, stop=total_distance, step=step).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curvature = np.zeros(len(t_array))
    achievement_rates = np.zeros(len(t_array))
    parts = np.zeros(len(t_array), dtype=object)
    i_end = len(t_array)

    for i, t in enumerate(t_array):
        if t > total_distance:
            i_end = i
            break

        if t <= circle_AB:
            t1 = t
            alpha = t1 / circle_AB
            t1_rad = (1 - alpha) * theta_1_start + alpha * theta_1_end
            x[i] = center_point1[0] + r1 * np.cos(t1_rad)
            y[i] = center_point1[1] + r1 * np.sin(t1_rad)
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

        elif circle_AB < t <= circle_AB + BC:
            t2 = t - circle_AB
            alpha = t2 / BC
            x[i] = (1 - alpha) * B[0] + alpha * C[0]
            y[i] = (1 - alpha) * B[1] + alpha * C[1]
            achievement_rates[i] = t / total_distance
            parts[i] = "linear"

        elif circle_AB + BC < t <= total_distance:
            t3 = t - (circle_AB + BC)
            alpha = t3 / circle_CD
            t3_rad = (1 - alpha) * theta_2_start + alpha * theta_2_end
            x[i] = center_point2[0] + r2 * np.cos(t3_rad)
            y[i] = center_point2[1] + r2 * np.sin(t3_rad)
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    delta_x = (
        amplitude
        * np.sin(8.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * (-np.sin(yaw))
    )
    delta_y = (
        amplitude
        * np.sin(8.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * (np.cos(yaw))
    )
    x += delta_x
    y += delta_y

    dx = (x[1:] - x[:-1]) / step
    dy = (y[1:] - y[:-1]) / step

    ddx = (dx[1:] - dx[:-1]) / step
    ddy = (dy[1:] - dy[:-1]) / step

    yaw = np.arctan2(dy, dx)
    yaw = np.array(yaw.tolist() + [yaw[-1]])

    curvature = 1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
    curvature = np.array(curvature.tolist() + [curvature[-2], curvature[-1]])
    
    yaw = yaw[:i_end]
    curvature = curvature[:i_end]

    return np.array([x, y]).T, yaw, curvature, parts, achievement_rates, total_distance


def calc_two_circles_excircle_trajectory(
    center_point1,
    r1,
    theta_1_start,
    theta_1_end,
    center_point2,
    r2,
    theta_2_start,
    theta_2_end,
    excircle_center_point,
    ext_r,
    theta_excircle_start,
    theta_excircle_end,
    step,
    amplitude=0.001,
):
    # A = [
    #    center_point1[0] + r1 * np.cos(theta_1_start),
    #    center_point1[1] + r1 * np.sin(theta_1_start),
    # ]
    # B = [center_point1[0] + r1 * np.cos(theta_1_end), center_point1[1] + r1 * np.sin(theta_1_end)]
    # C = [
    #    center_point2[0] + r2 * np.cos(theta_2_start),
    #    center_point2[1] + r2 * np.sin(theta_2_start),
    # ]

    circle_AB = r1 * abs(theta_1_end - theta_1_start)
    circle_BC = ext_r * abs(theta_excircle_end - theta_excircle_start)
    circle_CD = r2 * abs(theta_2_end - theta_2_start)

    total_distance = circle_AB + circle_BC + circle_CD
    t_array = np.arange(start=0.0, stop=total_distance, step=step).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curvature = np.zeros(len(t_array))
    achievement_rates = np.zeros(len(t_array))
    parts = np.zeros(len(t_array), dtype=object)

    i_end = len(t_array)

    for i, t in enumerate(t_array):
        if t > total_distance:
            i_end = i
            break

        if t <= circle_AB:
            t1 = t
            alpha = t1 / circle_AB
            t1_rad = (1 - alpha) * theta_1_start + alpha * theta_1_end
            x[i] = center_point1[0] + r1 * np.cos(t1_rad)
            y[i] = center_point1[1] + r1 * np.sin(t1_rad)
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

        elif circle_AB < t <= circle_AB + circle_BC:
            kappa = 0.0 / ext_r
            t2 = t - circle_AB
            alpha = t2 / circle_BC
            t2_rad = (1 - alpha) * theta_excircle_start + alpha * theta_excircle_end
            x[i] = excircle_center_point[0] + (
                1.0 + kappa * 4 * alpha * (1 - alpha) * 4 * alpha * (1 - alpha)
            ) * ext_r * np.cos(t2_rad)
            y[i] = excircle_center_point[1] + (
                1.0 + kappa * 4 * alpha * (1 - alpha) * 4 * alpha * (1 - alpha)
            ) * ext_r * np.sin(t2_rad)
            achievement_rates[i] = t / total_distance
            parts[i] = "linear"

        elif circle_AB + circle_BC < t <= total_distance:
            t3 = t - (circle_AB + circle_BC)
            alpha = t3 / circle_CD
            t3_rad = (1 - alpha) * theta_2_start + alpha * theta_2_end
            x[i] = center_point2[0] + r2 * np.cos(t3_rad)
            y[i] = center_point2[1] + r2 * np.sin(t3_rad)
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    delta_x = (
        amplitude
        * np.sin(8.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * (-np.sin(yaw))
    )
    delta_y = (
        amplitude
        * np.sin(8.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * np.sin(2.0 * np.pi * achievement_rates)
        * (np.cos(yaw))
    )
    x += delta_x
    y += delta_y

    dx = (x[1:] - x[:-1]) / step
    dy = (y[1:] - y[:-1]) / step

    ddx = (dx[1:] - dx[:-1]) / step
    ddy = (dy[1:] - dy[:-1]) / step

    yaw = np.arctan2(dy, dx)
    yaw = np.array(yaw.tolist() + [yaw[-1]])


    curvature = 1e-9 + abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1] ** 2 + dy[:-1] ** 2 + 1e-9) ** 1.5
    curvature = np.array(curvature.tolist() + [curvature[-2], curvature[-1]])
    yaw = yaw[:i_end]
    curvature = curvature[:i_end]

    return np.array([x, y]).T, yaw, curvature, parts, achievement_rates, total_distance


class TrajectorySegment:
    def __init__(self, 
                 segment_type: str, 
                 trajectory, 
                 yaw, 
                 curvature, 
                 parts, 
                 achievement_rates, 
                 in_direction: str, 
                 out_direction: str, 
                 reversing: bool,
                 distance: float):
        
        # Validation for segment_type
        if segment_type not in ["boundary", "non_boundary"]:
            raise ValueError("segment_type must be 'boundary' or 'non_boundary'")
        
        # Validation for in_direction and out_direction
        if in_direction not in ["clock_wise", "counter_clock_wise"]:
            raise ValueError("in_direction must be 'clock_wise' or 'counter_clock_wise'")
        if out_direction not in ["clock_wise", "counter_clock_wise"]:
            raise ValueError("out_direction must be 'clock_wise' or 'counter_clock_wise'")
        
        self.type = segment_type
        self.trajectory = trajectory
        self.yaw = yaw
        self.curvature = curvature
        self.parts = parts
        self.achievement_rates = achievement_rates
        self.in_direction = in_direction
        self.out_direction = out_direction
        self.reversing = reversing
        self.distance = distance

    def __repr__(self):
        return (f"TrajectorySegment(type={self.type}, trajectory={self.trajectory}, yaw={self.yaw}, "
                f"curvature={self.curvature}, parts={self.parts}, achievement_rates={self.achievement_rates}, "
                f"in_direction={self.in_direction}, out_direction={self.out_direction}, reversing={self.reversing}), distance={self.distance}")
    

def calc_two_circles_common_tangent_trajectory(R, r, step, amplitude=0.001):
    if r < R:
        center_1 = [-(R - r), 0.0]
        center_2 = [(R - r), 0.0]
        tangents_and_points = calc_tangents_and_points(
            center_1[0], center_1[1], r, center_2[0], center_2[1], r
        )
        for tangent_and_point in tangents_and_points:
            contact_point1 = tangent_and_point["contact_point1"]
            contact_point2 = tangent_and_point["contact_point2"]
            # 接線のうち、特定の条件に基づいて軌道を計算
            if contact_point1[1] > 0.0 and contact_point2[1] < 0.0 and r < R / 2:  # ここで条件を調整
                theta_1_end = np.arctan2(
                    contact_point1[1] - center_1[1], contact_point1[0] - center_1[0]
                )
                theta_2_start = np.arctan2(
                    contact_point2[1] - center_2[1], contact_point2[0] - center_2[0]
                )

                (
                    trajectory,
                    yaw,
                    curvature,
                    parts,
                    achievement_rates,
                    total_distance,
                ) = calc_two_circles_one_line_trajectory(
                    center_1,
                    r,
                    np.pi,
                    theta_1_end,
                    center_2,
                    r,
                    theta_2_start,
                    0.0,
                    step=step,
                    amplitude=amplitude,
                )

                return TrajectorySegment(
                    "non_boundary",
                    trajectory,
                    yaw,
                    curvature,
                    parts,
                    achievement_rates,
                    "clock_wise",
                    "counter_clock_wise",
                    True,
                    total_distance,
                )

            if contact_point1[1] > 0.0 and contact_point2[1] > 0.0 and R / 2 <= r < R:  # ここで条件を調整
                theta_1_end = np.arctan2(
                    contact_point1[1] - center_1[1], contact_point1[0] - center_1[0]
                )
                theta_2_start = np.arctan2(
                    contact_point2[1] - center_2[1], contact_point2[0] - center_2[0]
                )

                (
                    trajectory,
                    yaw,
                    curvature,
                    parts,
                    achievement_rates,
                    total_distance,
                ) = calc_two_circles_one_line_trajectory(
                    center_1,
                    r,
                    np.pi,
                    theta_1_end,
                    center_2,
                    r,
                    theta_2_start,
                    0.0,
                    step=step,
                    amplitude=amplitude,
                )

                return TrajectorySegment(
                    "non_boundary",
                    trajectory,
                    yaw,
                    curvature,
                    parts,
                    achievement_rates,
                    "clock_wise",
                    "clock_wise",
                    False,
                    total_distance,
                )
    elif R <= r:
        center_1 = [-(R - R * 0.75), 0.0]
        center_2 = [(R - R * 0.75), 0.0]
        (
            theta,
            (x_center, y_center),
            (contact_1_x, contact_1_y),
            (contact_2_x, contact_2_y),
        ) = calc_excircle_tangent_to_two_circles(
            center_1[0], center_1[1], center_2[0], center_2[1], R * 0.75, r
        )
        theta_1_end = np.arctan2(contact_1_y - center_1[1], contact_1_x - center_1[0])
        theta_2_start = np.arctan2(contact_2_y - center_2[1], contact_2_x - center_2[0])
        (
            trajectory,
            yaw,
            curvature,
            parts,
            achievement_rates,
            total_distance,
        ) = calc_two_circles_excircle_trajectory(
            center_point1=center_1,
            r1=R * 0.75,
            theta_1_start=np.pi,
            theta_1_end=theta_1_end,
            center_point2=center_2,
            r2=R * 0.75,
            theta_2_start=theta_2_start,
            theta_2_end=0.0,
            excircle_center_point=[x_center, y_center],
            ext_r=r,
            theta_excircle_start=np.pi - (np.pi - theta) / 2,
            theta_excircle_end=(np.pi - theta) / 2,
            step=step,
            amplitude=amplitude,
        )
        return TrajectorySegment(
            "non_boundary",
            trajectory,
            yaw,
            curvature,
            parts,
            achievement_rates,
            "clock_wise",
            "clock_wise",
            False,
            total_distance,
        )

    return None


def calc_boundary_trajectory(R):
    delta_theta = 0.6
    circle = R * delta_theta

    total_distance = circle
    t_array = np.arange(start=0.0, stop=total_distance, step=0.10).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curvature = np.zeros(len(t_array))
    achievement_rates = np.zeros(len(t_array))
    parts = np.zeros(len(t_array), dtype=object)

    i_end = len(t_array)

    theta_start = np.pi
    theta_end = np.pi - delta_theta

    for i, t in enumerate(t_array):
        if t > total_distance:
            i_end = i
            break

        if t <= circle:
            t1 = t
            alpha = t1 / circle
            t1_rad = (1 - alpha) * theta_start + alpha * theta_end
            x[i] = R * np.cos(t1_rad)
            y[i] = R * np.sin(t1_rad)
            tangent_x = (
                np.cos(np.pi / 2 + t1_rad)
                * (theta_end - theta_start)
                / abs(theta_end - theta_start)
            )
            tangent_y = (
                np.sin(np.pi / 2 + t1_rad)
                * (theta_end - theta_start)
                / abs(theta_end - theta_start)
            )
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curvature[i] = 1 / R
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    yaw = yaw[:i_end]
    curvature = curvature[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    return TrajectorySegment(
        "boundary",
        np.array([x, y]).T,
        yaw,
        curvature,
        parts,
        achievement_rates,
        "clock_wise",
        "clock_wise",
        False,
        total_distance,
    )


def reverse_trajectory_segment(trajectory_segment):
    Rot = np.array([[1.0, 0.0], [0.0, -1.0]])

    in_direction_reversed = "clock_wise"
    out_direction_reversed = "clock_wise"

    if trajectory_segment.in_direction == "clock_wise":
        in_direction_reversed = "counter_clock_wise"
    if trajectory_segment.out_direction == "clock_wise":
        out_direction_reversed = "counter_clock_wise"


    trajectory_segment_reversed = copy.deepcopy(trajectory_segment)
    trajectory_segment_reversed.trajectory = trajectory_segment.trajectory @ Rot
    trajectory_segment_reversed.yaw = -trajectory_segment.yaw
    trajectory_segment_reversed.in_direction = in_direction_reversed
    trajectory_segment_reversed.out_direction = out_direction_reversed

    return trajectory_segment_reversed


dead_band_delta_acc = 0.01
dead_band_delta_vel = dead_band_delta_acc * 0.03
delta_vel_gain = 0.5 / 4
delta_acc_gain = 20.0 / 4

class Reversal_Loop_Circle(Base_Course):
    def __init__(self, step: float, param_dict, R = 35.0):
        super().__init__(step, param_dict)
        self.closed = False

        self.R = R
        L = 2.79 # base link length
        self.acc_hist = deque([float(0.0)] * 10, maxlen=10)
        self.vel_hist = deque([float(0.0)] * 10, maxlen=10)
        self.previous_updated_time = 0.0

        self.previous_target_vel = 0.0
        self.target_vel_on_line = 0.0
        self.target_acc_on_line = 0.0
        self.vel_idx, self.acc_idx = 0, 0

        self.on_line_vel_flag = False
        self.prev_part = "left_circle"
        self.deceleration_rate = 0.70

        self.steer_list = [0.001, 0.01, 0.02, 0.03]

        self.amplitude_list = []
        self.inversion_delta_list = []

        self.steer_trajectory_clock_wise = {}
        self.steer_trajectory_counter_clock_wise = {}
        for i in range(len(self.steer_list)):

            r = L / np.tan(self.steer_list[i])
            trajectory = calc_two_circles_common_tangent_trajectory(self.R, r, step=self.step)
            self.steer_trajectory_clock_wise[self.steer_list[i]] = trajectory
            self.steer_trajectory_counter_clock_wise[self.steer_list[i]] = reverse_trajectory_segment(trajectory)

        self.inversion_trajectory = {}
        for i in range(100):

            amplitude = 0.1 * i
            self.amplitude_list.append(str(amplitude))

            inversion_trajectory__ = {}
            r = self.R / 2 * 0.75
            trajectory = calc_two_circles_common_tangent_trajectory(self.R, r, step=self.step, amplitude=amplitude)
            inversion_trajectory__["clock_wise_to_counter_clock_wise"] = trajectory

            trajectory_reversed = reverse_trajectory_segment(trajectory)
            inversion_trajectory__["counter_clock_wise_to_clock_wise"] = trajectory_reversed

            self.inversion_trajectory[str(amplitude)] = inversion_trajectory__
            self.inversion_delta_list.append(np.arctan2(np.max(trajectory_reversed.curvature) * 2.79, 1.0))

        self.boundary_trajectory = {}
        trajectory = calc_boundary_trajectory(self.R)
        self.boundary_trajectory["clock_wise"] = trajectory

        trajectory_reversed = reverse_trajectory_segment(trajectory)
        self.boundary_trajectory["counter_clock_wise"] = trajectory_reversed

        self.trajectory_list = [self.boundary_trajectory["clock_wise"]]
        self.end_point = self.trajectory_list[-1].trajectory[-1]
        self.latest_direction = "clock_wise"
        self.trajectory_length_list = [len(self.trajectory_list[-1].trajectory)]

        while len(self.trajectory_length_list) < 3:
                self.add_steer_trajectory()
                self.add_steer_trajectory()
                self.add_inversion_trajectory(self.steer_list[0])


    def add_trajectory(self, trajectory):

        if (
            self.trajectory_list[-1].type == "non_boundary"
            and trajectory.type == "non_boundary"
        ):
            self.add_trajectory( self.boundary_trajectory[self.latest_direction] )

        if (
            trajectory.in_direction is not self.latest_direction
            and self.trajectory_list[-1].type == "boundary"
        ):
            key_ = self.latest_direction + "_to_" + getattr(trajectory, "in_direction")
            self.add_trajectory(self.inversion_trajectory[self.amplitude_list[0]][key_])

        if (
            self.trajectory_list[-1].type == "non_boundary"
            and trajectory.type == "non_boundary"
        ):
            self.add_trajectory( self.boundary_trajectory[self.latest_direction] )

        cos = self.end_point[0] / self.R
        sin = self.end_point[1] / self.R
        delta_theta = np.arctan2(sin, cos) - np.pi
        Rotation = (-np.eye(2)) @ np.array([[cos, -sin], [sin, cos]])
        
        trajectory_modified = copy.deepcopy(trajectory)
        trajectory_modified.trajectory = (Rotation @ trajectory.trajectory.T).T
        trajectory_modified.yaw = trajectory.yaw + delta_theta
        
        self.latest_direction = trajectory.out_direction
        self.end_point = trajectory_modified.trajectory[-1]

        self.trajectory_list.append(trajectory_modified)
        self.trajectory_length_list.append(len(trajectory_modified.trajectory))

    def add_steer_trajectory(self):
        if self.latest_direction == "clock_wise":
            self.add_trajectory(
                self.steer_trajectory_clock_wise[random.choice(self.steer_list)]
            )
        elif self.latest_direction == "counter_clock_wise":
            self.add_trajectory(
                self.steer_trajectory_counter_clock_wise[random.choice(self.steer_list)]
            )

    def remove_trajectory(self):
        self.trajectory_list.pop(0)
        self.trajectory_length_list.pop(0)

    def add_inversion_trajectory(self, steer):
        index = (np.abs(np.array(self.inversion_delta_list) - steer)).argmin()

        if self.latest_direction == "clock_wise":
            self.add_trajectory(
                self.inversion_trajectory[self.amplitude_list[index]][
                    self.latest_direction + "_to_" + "counter_clock_wise"
                ]
            )
        elif self.latest_direction == "counter_clock_wise":
            self.add_trajectory(
                self.inversion_trajectory[self.amplitude_list[index]][
                    self.latest_direction + "_to_" + "clock_wise"
                ]
            )

    def get_trajectory_points(
        self,
        long_side_length: float,
        short_side_length: float,
        ego_point=np.array([0.0, 0.0]),
        goal_point=np.array([0.0, 0.0]),
    ):
        self.trajectory_points = np.concatenate(
            [self.trajectory_list[i].trajectory for i in range(4)],
            axis=0,
        )
        self.yaw = np.concatenate(
            [self.trajectory_list[i].yaw for i in range(4)],
            axis=0,
        )
        self.parts = np.concatenate(
            [self.trajectory_list[i].parts for i in range(4)], axis=0
        )
        self.curvature = np.concatenate(
            [self.trajectory_list[i].curvature for i in range(4)], axis=0
        )
        self.achievement_rates = np.concatenate(
            [self.trajectory_list[i].achievement_rates for i in range(4)], axis=0
        )
    
    def get_target_velocity(self, nearestIndex, current_time, current_vel, current_acc, collected_data_counts_of_vel_acc
    ):  
        max_curvature_on_segment = np.max(self.trajectory_list[1].curvature)
        max_curvature_on_segment = np.max(
            [max_curvature_on_segment, np.max(self.trajectory_list[2].curvature)]
        )

        max_vel_from_lateral_acc = np.sqrt( self.params.max_lateral_accel  / max_curvature_on_segment )
        T = 10.0
        if (
            current_time - self.previous_updated_time > 3.00 * T
            or collected_data_counts_of_vel_acc[self.vel_idx, self.acc_idx]
            > np.max([50, np.mean(np.minimum(collected_data_counts_of_vel_acc, 200))])  or self.target_vel_on_line > max_vel_from_lateral_acc
        ):
            min_data_num_margin = 20
            min_index_list = []

            min_num_data = 1e12
            N_V = self.params.num_bins_v
            N_A = self.params.num_bins_a

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
                        if min_num_data - min_data_num_margin > collected_data_counts_of_vel_acc[i, j]:
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
            self.acc_on_line = self.target_acc_on_line

            self.previous_updated_time = current_time
            self.near_target_vel = False
            
        rate__ = self.achievement_rates[nearestIndex]
        self.acc_hist.append(float(current_acc))

        if abs(current_vel - np.max([(self.target_vel_on_line - 1.0 * self.target_acc_on_line), 0.05]) ) <   0.1 and not self.near_target_vel:
            self.near_target_vel = True
            self.phase_shift = current_time - self.previous_updated_time

        if current_time - self.previous_updated_time < 1.0 * T and not self.near_target_vel:

            self.phase_shift = current_time - self.previous_updated_time

            delta_vel =   (current_vel - np.max([(self.target_vel_on_line - 1.0 * self.target_acc_on_line),0.05]))
            if abs(delta_vel) < dead_band_delta_vel:
                delta_vel = 0.0 

            target_vel = np.max([(self.target_vel_on_line - 1.0 * self.target_acc_on_line), 0.05]) + np.clip(-(delta_vel_gain * delta_vel)**3, -0.5, 0.5)#- self.target_acc_on_line / 2
            target_vel = np.max([target_vel, 0.05])

            self.vel_hist.append(target_vel)
            target_vel = np.mean(self.vel_hist)
        else:
            if current_vel < np.max([(self.target_vel_on_line - 1.0 * abs(self.target_acc_on_line)), 0.05]):
                self.acc_on_line = abs(self.target_acc_on_line)
            elif current_vel > self.target_vel_on_line + 1.0 * abs(self.target_acc_on_line):
                self.acc_on_line = -abs(self.target_acc_on_line)
            
            delta_acc = current_acc - self.acc_on_line
            if abs(delta_acc) < dead_band_delta_acc:
                delta_acc = 0.0
            target_vel = current_vel + np.clip(self.acc_on_line / 1.0 -(delta_acc_gain *  delta_acc)**3, -1.0, 1.0)

        target_vel = np.clip(target_vel, self.previous_target_vel - 0.03 * 2.0, self.previous_target_vel + 0.03 * 2.0)
        target_vel = np.min([max_vel_from_lateral_acc, target_vel])

        if self.trajectory_list[0].reversing == True or self.trajectory_list[1].reversing == True:
            target_vel = 3.0 + 3.0 * np.sin(0.25 * np.pi * current_time / T) * np.sin(0.5 * np.pi * current_time / T)

            if self.trajectory_list[0].reversing == True:
                target_vel = 3.0

        target_vel = np.max([target_vel, 0.05])

        self.previous_target_vel = target_vel

        return target_vel
    
    def update_trajectory_points(self, nearestIndex, yaw_offset, rectangle_center_position):
        if nearestIndex > self.trajectory_length_list[0] + self.trajectory_length_list[1]:
            self.remove_trajectory()

            while len(self.trajectory_length_list) < 4:
                self.add_steer_trajectory()
                self.add_steer_trajectory()
                self.add_inversion_trajectory(self.steer_list[0])

        self.trajectory_points = np.concatenate(
            [self.trajectory_list[i].trajectory for i in range(4)],
            axis=0,
        )
        self.yaw = np.concatenate(
            [self.trajectory_list[i].yaw for i in range(4)],
            axis=0,
        )
        self.parts = np.concatenate(
            [self.trajectory_list[i].parts for i in range(4)], axis=0
        )
        self.curvature = np.concatenate(
            [self.trajectory_list[i].curvature for i in range(4)], axis=0
        )
        self.achievement_rates = np.concatenate(
            [self.trajectory_list[i].achievement_rates for i in range(4)], axis=0
        )

        return self.return_trajectory_points(yaw_offset, rectangle_center_position)

    def get_boundary_points(self):
        if self.trajectory_points is None or self.yaw is None:
            return None

        outer_circle_radius = self.R + 5.0
        theta_list = np.linspace(0, 2 * np.pi, 100)
        boundary_points = []
        
        center_point_x = (self.A[0] + self.B[0] + self.C[0] + self.D[0]) / 4
        center_point_y = (self.A[1] + self.B[1] + self.C[1] + self.D[1]) / 4

        for theta in theta_list:
            point_x = outer_circle_radius * np.cos(theta) + center_point_x
            point_y = outer_circle_radius * np.sin(theta) + center_point_y
            boundary_points.append([point_x, point_y])

        self.boundary_points = np.array(boundary_points)
