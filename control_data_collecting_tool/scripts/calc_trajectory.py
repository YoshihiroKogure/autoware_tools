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
import numpy as np
import matplotlib.pyplot as plt

# acosの範囲外の値を処理するための関数
def safe_acos(value):
    return math.acos(max(-1, min(1, value)))  # -1から1の範囲にクリップ

# 接点の正しい計算を行う関数
def calc_tangent_contact_points(x, y, r, a, b, c):
    # 接線の方程式 ax + by = c を円の方程式に代入し、接点を求める
    if b != 0:
        # y = (c - a * x) / b を使って解く
        A = 1 + (a / b) ** 2
        B = -2 * (x + (a * (c - b * y)) / b**2)
        C = x**2 + ((c - b * y) / b)**2 - r**2

        # 判別式を計算
        discriminant = abs(B**2 - 4 * A * C)

        if discriminant < 0:
            return None  # 接点なし

        # 判別式に基づいて接点を計算
        x_contact1 = (-B + math.sqrt(discriminant)) / (2 * A)
        x_contact2 = (-B - math.sqrt(discriminant)) / (2 * A)

        y_contact1 = (c - a * x_contact1) / b
        y_contact2 = (c - a * x_contact2) / b

    else:
        # b == 0 の場合 (垂直な接線)
        x_contact1 = x_contact2 = c / a
        y_contact1 = y + math.sqrt(r**2 - (x_contact1 - x)**2)
        y_contact2 = y - math.sqrt(r**2 - (x_contact1 - x)**2)

    return [(x_contact1, y_contact1), (x_contact2, y_contact2)]

# 接線と接点を計算する関数
def calc_tangents_and_points(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    d_sq = dx**2 + dy**2
    d = math.sqrt(d_sq)

    if d == 0 or d < abs(r1 - r2):
        return []  # 中心が一致するか、円が入れ子状態では共通接線は存在しない

    tangents_and_points = []

    # 外接線の場合
    try:
        a = math.atan2(dy, dx)
        b = safe_acos((r1 - r2) / d)  # acosの引数が範囲外にならないように
        t1 = a + b
        t2 = a - b

        for t in [t1, t2]:
            cos_t = math.cos(t)
            sin_t = math.sin(t)

            a = cos_t
            b = sin_t
            c = a * x1 + b * y1 + r1

            points1 = calc_tangent_contact_points(x1, y1, r1, a, b, c)
            points2 = calc_tangent_contact_points(x2, y2, r2, a, b, c)
            if points1 is not None and points2 is not None:
                tangents_and_points.append({
                    'tangent': (a, b, c),
                    'contact_point1': points1[0],
                    'contact_point2': points2[0]
                })
    except ValueError:
        pass

    # 内接線の場合
    try:
        c = safe_acos((r1 + r2) / d)  # acosの引数が範囲外にならないように
        t3 = a + c
        t4 = a - c

        for t in [t3, t4]:
            cos_t = math.cos(t)
            sin_t = math.sin(t)

            a = cos_t
            b = sin_t
            c = a * x1 + b * y1 + r1

            points1 = calc_tangent_contact_points(x1, y1, r1, a, b, c)
            points2 = calc_tangent_contact_points(x2, y2, r2, a, b, c)

            if points1 is not None and points2 is not None:
                tangents_and_points.append({
                    'tangent': (a, b, c),
                    'contact_point1': points1[0],
                    'contact_point2': points2[0]
                })
    except ValueError:
        pass

    return tangents_and_points    

# 軌道計算関数
def calc_two_circles_one_line_trajectory(center_point1, r1, theta_1_start, theta_1_end,
                                        center_point2, r2, theta_2_start, theta_2_end,
                                        step):
    A = [center_point1[0] + r1 * np.cos(theta_1_start), center_point1[1] + r1 * np.sin(theta_1_start)]
    B = [center_point1[0] + r1 * np.cos(theta_1_end), center_point1[1] + r1 * np.sin(theta_1_end)]
    C = [center_point2[0] + r2 * np.cos(theta_2_start), center_point2[1] + r2 * np.sin(theta_2_start)]

    circle_AB = r1 * abs(theta_1_end - theta_1_start)
    circle_CD = r2 * abs(theta_2_end - theta_2_start)
    BC = np.linalg.norm(np.array(B) - np.array(C))

    total_distance = circle_AB + BC + circle_CD
    t_array = np.arange(start=0.0, stop=total_distance, step=step).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curve = np.zeros(len(t_array))
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
            tangent_x = np.cos(np.pi / 2 + t1_rad) * (theta_1_end - theta_1_start) / abs(theta_1_end - theta_1_start)
            tangent_y = np.sin(np.pi / 2 + t1_rad) * (theta_1_end - theta_1_start) / abs(theta_1_end - theta_1_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / r1
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

        elif circle_AB < t <= circle_AB + BC:
            t2 = t - circle_AB
            alpha = t2 / BC
            x[i] = (1 - alpha) * B[0] + alpha * C[0]
            y[i] = (1 - alpha) * B[1] + alpha * C[1]
            yaw[i] = -(circle_AB / r1 - np.pi / 2) #* (C[1] - B[1]) / abs(C[1] - B[1]) 
            curve[i] = 1e-9
            achievement_rates[i] = t / total_distance
            parts[i] = "linear"

        elif circle_AB + BC < t <= total_distance:
            t3 = t - (circle_AB + BC)
            alpha = t3 / circle_CD
            t3_rad = (1 - alpha) * theta_2_start + alpha * theta_2_end
            x[i] = center_point2[0] + r2 * np.cos(t3_rad)
            y[i] = center_point2[1] + r2 * np.sin(t3_rad)
            tangent_x = np.cos(np.pi / 2 + t3_rad) * (theta_2_end - theta_2_start) / abs(theta_2_end - theta_2_start)
            tangent_y = np.sin(np.pi / 2 + t3_rad) * (theta_2_end - theta_2_start) / abs(theta_2_end - theta_2_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / r2
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    yaw = yaw[:i_end]
    curve = curve[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    return np.array([x, y]).T, yaw, curve, parts, achievement_rates, total_distance



def calc_excircle_tangent_to_two_circles(x1, y1, x2, y2, r, r_ext):

    # 2つの円の中心間の距離
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d == 0:
        raise ValueError("同心円の場合、外接円は存在しません。")
    
    # 外接円の中心を計算
    R_r = r_ext - r
    theta = np.arctan2(np.sqrt(R_r**2 - (d/2)**2 ),d/2)  # 2つの円の中心を結ぶ角度
    
     # 外接円の中心の座標
    x_center = (x2 - R_r * np.cos(theta) + x1 - R_r * np.cos(np.pi - theta)) / 2
    y_center = (y2 - R_r * np.sin(theta) + y1 - R_r * np.sin(np.pi - theta)) / 2

    # 各円の接点の座標を計算
    contact_1_x = x1 + r * np.cos(np.pi - theta)
    contact_1_y = y1 + r * np.sin(np.pi - theta)

    contact_2_x = x2 + r * np.cos(theta)
    contact_2_y = y2 + r * np.sin(theta)
    
    return np.pi - 2 * theta, [x_center, y_center], [contact_1_x, contact_1_y], [contact_2_x, contact_2_y]


# 軌道計算関数
def calc_two_circles_excircle_trajectory(center_point1, r1, theta_1_start, theta_1_end,
                                        center_point2, r2, theta_2_start, theta_2_end,
                                        excircle_center_point, ext_r, theta_excircle_start, theta_excircle_end,
                                        step):
    A = [center_point1[0] + r1 * np.cos(theta_1_start), center_point1[1] + r1 * np.sin(theta_1_start)]
    B = [center_point1[0] + r1 * np.cos(theta_1_end), center_point1[1] + r1 * np.sin(theta_1_end)]
    C = [center_point2[0] + r2 * np.cos(theta_2_start), center_point2[1] + r2 * np.sin(theta_2_start)]

    circle_AB = r1 * abs(theta_1_end - theta_1_start)
    circle_BC = ext_r * abs(theta_excircle_end - theta_excircle_start)
    circle_CD = r2 * abs(theta_2_end - theta_2_start)

    total_distance = circle_AB + circle_BC + circle_CD
    t_array = np.arange(start=0.0, stop=total_distance, step=step).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curve = np.zeros(len(t_array))
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
            tangent_x = np.cos(np.pi / 2 + t1_rad) * (theta_1_end - theta_1_start) / abs(theta_1_end - theta_1_start)
            tangent_y = np.sin(np.pi / 2 + t1_rad) * (theta_1_end - theta_1_start) / abs(theta_1_end - theta_1_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / r1
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

        elif circle_AB < t <= circle_AB + circle_BC:
            t2 = t - circle_AB
            alpha = t2 / circle_BC
            t2_rad = (1 - alpha) * theta_excircle_start + alpha * theta_excircle_end
            x[i] = excircle_center_point[0] + ext_r * np.cos(t2_rad)
            y[i] = excircle_center_point[1] + ext_r * np.sin(t2_rad)
            tangent_x = np.cos(np.pi / 2 + t2_rad) * (theta_excircle_end - theta_excircle_start) / abs(theta_excircle_end - theta_excircle_start)
            tangent_y = np.sin(np.pi / 2 + t2_rad) * (theta_excircle_end - theta_excircle_start) / abs(theta_excircle_end - theta_excircle_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / ext_r
            achievement_rates[i] = t / total_distance
            parts[i] = "linear"

        elif circle_AB + circle_BC < t <= total_distance:
            t3 = t - (circle_AB + circle_BC)
            alpha = t3 / circle_CD
            t3_rad = (1 - alpha) * theta_2_start + alpha * theta_2_end
            x[i] = center_point2[0] + r2 * np.cos(t3_rad)
            y[i] = center_point2[1] + r2 * np.sin(t3_rad)
            tangent_x = np.cos(np.pi / 2 + t3_rad) * (theta_2_end - theta_2_start) / abs(theta_2_end - theta_2_start)
            tangent_y = np.sin(np.pi / 2 + t3_rad) * (theta_2_end - theta_2_start) / abs(theta_2_end - theta_2_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / r2
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    yaw = yaw[:i_end]
    curve = curve[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    return np.array([x, y]).T, yaw, curve, parts, achievement_rates, total_distance


def calc_two_circles_common_tangent_trajectory(R,r):
    
    if r < R:
        center_1 = [-(R - r), 0.0]
        center_2 = [ (R - r), 0.0]
        tangents_and_points = calc_tangents_and_points(center_1[0], center_1[1], r, center_2[0], center_2[1], r)
        for tangent_and_point in tangents_and_points:
            contact_point1 = tangent_and_point['contact_point1']
            contact_point2 = tangent_and_point['contact_point2']
            # 接線のうち、特定の条件に基づいて軌道を計算
            if contact_point1[1] > 0.0 and contact_point2[1] < 0.0 and r < R/2:  # ここで条件を調整
                theta_1_end = np.arctan2(contact_point1[1] - center_1[1], contact_point1[0] - center_1[0])
                theta_2_start = np.arctan2(contact_point2[1] - center_2[1], contact_point2[0] - center_2[0])

                # 軌道を計算
                traj, yaw, curve, parts, achievement_rates, total_distance = calc_two_circles_one_line_trajectory(
                    center_1, r, np.pi, theta_1_end, center_2, r, theta_2_start, 0.0, 0.05)

                return "non_boundary", traj, yaw, curve, parts, achievement_rates, "clock_wise", "counter_clock_wise", total_distance

            if contact_point1[1] > 0.0 and contact_point2[1] > 0.0 and R/2 <= r < R:  # ここで条件を調整
                theta_1_end = np.arctan2(contact_point1[1] - center_1[1], contact_point1[0] - center_1[0])
                theta_2_start = np.arctan2(contact_point2[1] - center_2[1], contact_point2[0] - center_2[0])

                # 軌道を計算
                traj, yaw, curve, parts, achievement_rates, total_distance = calc_two_circles_one_line_trajectory(
                    center_1, r, np.pi, theta_1_end, center_2, r, theta_2_start, 0.0, 0.05)

                return "non_boundary", traj, yaw, curve, parts, achievement_rates, "clock_wise","clock_wise", total_distance
    elif R <= r:
        center_1 = [-(R - R * 0.5), 0.0]
        center_2 = [ (R - R * 0.5), 0.0]
        theta, (x_center, y_center), (contact_1_x, contact_1_y), (contact_2_x, contact_2_y) = calc_excircle_tangent_to_two_circles(center_1[0], center_1[1], center_2[0], center_2[1], R * 0.5, r)
        theta_1_end = np.arctan2(contact_1_y - center_1[1], contact_1_x - center_1[0])
        theta_2_start = np.arctan2(contact_2_y - center_2[1], contact_2_x - center_2[0])
        traj, yaw, curve, parts, achievement_rates, total_distance = calc_two_circles_excircle_trajectory(center_point1 = center_1, 
                                                r1 = R * 0.5,
                                                theta_1_start = np.pi,
                                                theta_1_end = theta_1_end,
                                                center_point2 = center_2, 
                                                r2 = R * 0.5, 
                                                theta_2_start = theta_2_start, 
                                                theta_2_end = 0.0,
                                                excircle_center_point=[x_center, y_center], 
                                                ext_r = r, 
                                                theta_excircle_start = np.pi - (np.pi - theta) / 2, 
                                                theta_excircle_end = (np.pi - theta) / 2, 
                                                step = 0.1)
        return "non_boundary", traj, yaw, curve, parts, achievement_rates, "clock_wise", "clock_wise", total_distance
            
    return None

def calc_boundary_trajectory(R):

    delta_theta = 0.4
    circle = R * delta_theta

    total_distance = circle
    t_array = np.arange(start=0.0, stop=total_distance, step=0.1).astype("float")

    x = np.zeros(len(t_array))
    y = np.zeros(len(t_array))
    yaw = np.zeros(len(t_array))
    curve = np.zeros(len(t_array))
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
            tangent_x = np.cos(np.pi / 2 + t1_rad) * (theta_end - theta_start) / abs(theta_end - theta_start)
            tangent_y = np.sin(np.pi / 2 + t1_rad) * (theta_end - theta_start) / abs(theta_end - theta_start)
            yaw[i] = math.atan2(tangent_y, tangent_x)
            curve[i] = 1 / R
            achievement_rates[i] = t / total_distance
            parts[i] = "circle"

    x = x[:i_end]
    y = y[:i_end]
    yaw = yaw[:i_end]
    curve = curve[:i_end]
    parts = parts[:i_end]
    achievement_rates = achievement_rates[:i_end]

    return "boundary", np.array([x, y]).T, yaw, curve, parts, achievement_rates, "clock_wise", "clock_wise", total_distance 



class calc_trajectory:

    def __init__(self,R):

        Rot = np.array([[1.0,0.0],[0.0,-1.0]])
        self.R = R
        L = 2.79
        self.steer_list =[-1.0,-0.90,-0.80,-0.70,-0.60, -0.50, -0.40] + [0.40, 0.50, 0.60,0.70,0.80,0.90,1.0]
        # middle low
        self.steer_list =[-0.60, -0.50, -0.40] + [ -0.3 + 0.1*i for i in range(3)] + [0.1*(1+i) for i in range(3)] + [0.40, 0.50, 0.60]

        #self.steer_list =  [ -0.3 + 0.03*i for i in range(10)] + [0.03*(1+i) for i in range(10)]

        # middle high
        self.steer_list =  [ -0.15 + 0.03*i for i in range(5)] + [0.03*(1+i) for i in range(5)]

        #high 
        # self.steer_list =  [ -0.03 + 0.01* i for i in range(3)] + [0.01*(1+i) for i in range(3)]

        self.steer_trajectory = {}
        for i in range(int(len(self.steer_list)/2),len(self.steer_list)):
            r = L / np.tan(self.steer_list[i])
            boundary, traj, yaw, curve, parts, achievement_rates, in_direction, out_direction, total_distance = calc_two_circles_common_tangent_trajectory(self.R, r)
            
            trajectory = {"type":boundary, "trajectory":traj, "yaw":yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction":in_direction, "out_direction":out_direction, "total_distance":total_distance}
            self.steer_trajectory[self.steer_list[i]] = trajectory
            
            in_direction_reversed = "clock_wise"
            out_direction_reversed = "clock_wise"
            if in_direction == "clock_wise":
                in_direction_reversed = "counter_clock_wise"
            if out_direction == "clock_wise":
                out_direction_reversed = "counter_clock_wise"     

            trajectory_reversed = {"type":boundary, "trajectory":traj@Rot,"yaw":-yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction":in_direction_reversed, "out_direction":out_direction_reversed, "total_distance":total_distance}
            self.steer_trajectory[self.steer_list[len(self.steer_list)-1-i]] = trajectory_reversed


        self.inversion_trajectory = {}
        r = self.R / 2 * 0.95
        boundary, traj, yaw, curve, parts, achievement_rates, _, _, total_distance = calc_two_circles_common_tangent_trajectory(self.R, r)
        trajectory = {"type":boundary, "trajectory":traj,"yaw":yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction":"clock_wise", "out_direction":"counter_clock_wise", "total_distance":total_distance}
        self.inversion_trajectory["clock_wise_to_counter_clock_wise"] = trajectory

        trajectory_reversed = {"type":boundary, "trajectory":traj@Rot,"yaw":-yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction":"counter_clock_wise", "out_direction":"clock_wise", "total_distance":total_distance}
        self.inversion_trajectory["counter_clock_wise_to_clock_wise"] = trajectory_reversed


        self.boundary_trajectory = {}
        boundary, traj, yaw, curve, parts, achievement_rates, _, _, total_distance = calc_boundary_trajectory(self.R)
        trajectory = {"type":boundary, "trajectory":traj, "yaw":yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction":"clock_wise", "out_direction":"clock_wise", "total_distance":total_distance}
        self.boundary_trajectory["clock_wise"] = trajectory
        trajectory_reversed = {"type":boundary, "trajectory":traj@Rot,"yaw":-yaw, "curve":curve, "parts":parts, "achievement_rates":achievement_rates, "in_direction": "counter_clock_wise","out_direction":"counter_clock_wise", "total_distance":total_distance}
        self.boundary_trajectory["counter_clock_wise"] = trajectory_reversed

        self.trajectory_list = [ self.boundary_trajectory["clock_wise"] ]
        self.end_point = self.trajectory_list[-1]["trajectory"][-1]
        self.latest_direction = "clock_wise"
        self.trajectory_length_list = [ len( self.trajectory_list[-1]["trajectory"]) ]

    
    def add_trajectory(self, trajectory):
        print(trajectory["in_direction"])

        if self.trajectory_list[-1]["type"] == "non_boundary" and trajectory["type"] == "non_boundary":
            self.add_trajectory(self.boundary_trajectory[self.latest_direction])

        if trajectory["in_direction"] is not self.latest_direction and self.trajectory_list[-1]["type"] == "boundary":
            key_ = self.latest_direction + "_to_" + trajectory["in_direction"]
            self.add_trajectory(self.inversion_trajectory[key_])

        if self.trajectory_list[-1]["type"] == "non_boundary" and trajectory["type"] == "non_boundary":
            self.add_trajectory(self.boundary_trajectory[self.latest_direction])


        cos = self.end_point[0] / self.R
        sin = self.end_point[1] / self.R
        delta_theta = np.arctan2(sin,cos) - np.pi
        Rotation = (-np.eye(2))@np.array([[cos, -sin],[sin,cos]])

        modified_traj = (Rotation@trajectory["trajectory"].T).T
        modified_yaw = trajectory["yaw"] + delta_theta
        modified_trajectory = {"type":trajectory["type"], "trajectory":modified_traj, "yaw":modified_yaw, "curve":trajectory["curve"], "parts":trajectory["parts"], "achievement_rates":trajectory["achievement_rates"], "in_direction":trajectory["in_direction"], "out_direction":trajectory["out_direction"], "total_distance":trajectory["total_distance"]}
        self.latest_direction = trajectory["out_direction"]

        self.end_point = modified_traj[-1]

        self.trajectory_list.append(modified_trajectory)
        self.trajectory_length_list.append(len(modified_traj))

    def add_steer_trajectory(self, steer_index):
        self.add_trajectory( self.steer_trajectory[self.steer_list[steer_index]] )

    def remove_trajectory(self):
        self.trajectory_list.pop(0)
        self.trajectory_length_list.pop(0)