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

from autoware_planning_msgs.msg import Trajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from courses.load_course import load_course
from data_collecting_base_node import DataCollectingBaseNode
from geometry_msgs.msg import Point
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos
from numpy import pi
from numpy import sin
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

debug_matplotlib_plot_flag = False
Differential_Smoothing_Flag = True
USE_CURVATURE_RADIUS_FLAG = False


def smooth_bounding(upper: np.ndarray, threshold: np.ndarray, x: np.ndarray):
    result = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if x[i] <= threshold[i]:
            result[i] = x[i]
        else:
            z = np.exp(-(x[i] - threshold[i]) / (upper[i] - threshold[i]))
            result[i] = upper[i] * (1 - z) + threshold[i] * z
    return result


def getYaw(orientation_xyzw):
    return R.from_quat(orientation_xyzw.reshape(-1, 4)).as_euler("xyz")[:, 2]


def computeTriangleArea(A, B, C):
    return 0.5 * abs(np.cross(B - A, C - A))


# inherits from DataCollectingBaseNode
class DataCollectingTrajectoryPublisher(DataCollectingBaseNode):
    def __init__(self):
        super().__init__("data_collecting_trajectory_publisher")

        self.declare_parameter(
            "COURSE_NAME",
            "eight_course",
            ParameterDescriptor(
                description="Course name [eight_course, u_shaped_return, straight_line_positive, straight_line_negative]"
            ),
        )

        self.declare_parameter(
            "acc_kp",
            1.0,
            ParameterDescriptor(description="Pure pursuit accel command proportional gain"),
        )

        self.declare_parameter(
            "max_lateral_accel",
            0.5,
            ParameterDescriptor(description="Max lateral acceleration limit [m/ss]"),
        )

        self.declare_parameter(
            "lateral_error_threshold",
            2.0,
            ParameterDescriptor(
                description="Lateral error threshold where applying velocity limit [m/s]"
            ),
        )

        self.declare_parameter(
            "yaw_error_threshold",
            0.50,
            ParameterDescriptor(
                description="Yaw error threshold where applying velocity limit [rad]"
            ),
        )

        self.declare_parameter(
            "velocity_limit_by_tracking_error",
            1.0,
            ParameterDescriptor(
                description="Velocity limit when tracking error exceeds threshold [m/s]"
            ),
        )

        self.declare_parameter(
            "mov_ave_window",
            50,
            ParameterDescriptor(description="Moving average smoothing window size"),
        )

        self.declare_parameter(
            "target_longitudinal_velocity",
            6.0,
            ParameterDescriptor(description="Target longitudinal velocity [m/s]"),
        )

        self.declare_parameter(
            "longitudinal_velocity_noise_amp",
            0.01,
            ParameterDescriptor(
                description="Target longitudinal velocity additional sine noise amplitude [m/s]"
            ),
        )

        self.declare_parameter(
            "longitudinal_velocity_noise_min_period",
            5.0,
            ParameterDescriptor(
                description="Target longitudinal velocity additional sine noise minimum period [s]"
            ),
        )

        self.declare_parameter(
            "longitudinal_velocity_noise_max_period",
            20.0,
            ParameterDescriptor(
                description="Target longitudinal velocity additional sine noise maximum period [s]"
            ),
        )

        self.declare_parameter(
            "COLLECTING_DATA_V_MIN",
            0.0,
            ParameterDescriptor(description="Minimum velocity for data collection [m/s]"),
        )

        self.declare_parameter(
            "COLLECTING_DATA_V_MAX",
            11.5,
            ParameterDescriptor(description="Maximum velocity for data collection [m/s]"),
        )

        self.declare_parameter(
            "COLLECTING_DATA_A_MIN",
            -1.0,
            ParameterDescriptor(description="Minimum velocity for data collection [m/ss]"),
        )

        self.declare_parameter(
            "COLLECTING_DATA_A_MAX",
            1.0,
            ParameterDescriptor(description="Maximum velocity for data collection [m/ss]"),
        )

        self.trajectory_for_collecting_data_pub_ = self.create_publisher(
            Trajectory,
            "/data_collecting_trajectory",
            1,
        )

        self.data_collecting_trajectory_marker_array_pub_ = self.create_publisher(
            MarkerArray,
            "/data_collecting_trajectory_marker_array",
            1,
        )

        self.pub_stop_request_ = self.create_publisher(
            Bool,
            "/data_collecting_stop_request",
            1,
        )

        self.sub_data_collecting_area_ = self.create_subscription(
            PolygonStamped,
            "/data_collecting_area",
            self.onDataCollectingArea,
            1,
        )
        self.sub_data_collecting_area_

        self.ego_point = np.array([0.0, 0.0])
        self.goal_point = np.array([0.0, 0.0])
        self.sub_data_collecting_gaol_pose_ = self.create_subscription(
            PoseStamped,
            "/data_collecting_goal_pose",
            self.onGoalPose,
            1,
        )
        self.sub_data_collecting_area_

        # obtain ros params as dictionary
        param_names = self._parameters
        params = self.get_parameters(param_names)
        params_dict = {param.name: param.value for param in params}

        # set course name
        self.COURSE_NAME = self.get_parameter("COURSE_NAME").value
        self.traj_step = 0.1
        self.course = load_course(self.COURSE_NAME , self.traj_step, params_dict)

        self.timer_period_callback = 0.03  # 30ms
        self.timer_traj = self.create_timer(self.timer_period_callback, self.timer_callback_traj)

        if debug_matplotlib_plot_flag:
            self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 20))
            plt.ion()

        self._present_kinematic_state = None
        self._present_acceleration = None

        self.trajectory_position_data = None
        self.trajectory_yaw_data = None
        self.trajectory_longitudinal_velocity_data = None
        self.trajectory_curvature_data = None
        self.trajectory_parts = None
        self.trajectory_achievement_rates = None

        self.current_target_longitudinal_velocity = (
            self.get_parameter("target_longitudinal_velocity").get_parameter_value().double_value
        )
        self.current_window = (
            self.get_parameter("mov_ave_window").get_parameter_value().integer_value
        )

        self.one_round_progress_rate = None
        self.vel_noise_list = []

        # subscriptions of data counter
        self.collected_data_counts_of_vel_acc_subscription_ = self.create_subscription(
            Int32MultiArray,
            "/control_data_collecting_tools/collected_data_counts_of_vel_acc",
            self.subscribe_collected_data_counts_of_vel_acc,
            10,
        )
        self.collected_data_counts_of_vel_acc_subscription_

        self.collected_data_counts_of_vel_steer_subscription_ = self.create_subscription(
            Int32MultiArray,
            "/control_data_collecting_tools/collected_data_counts_of_vel_steer",
            self.subscribe_collected_data_counts_of_vel_steer,
            10,
        )
        self.collected_data_counts_of_vel_steer_subscription_

        self.collected_data_counts_of_vel_steer_subscription_ = self.create_subscription(
            Int32MultiArray,
            "/control_data_collecting_tools/collected_data_counts_of_vel_steer",
            self.subscribe_collected_data_counts_of_vel_steer,
            10,
        )
        self.collected_data_counts_of_vel_steer_subscription_

    def subscribe_collected_data_counts_of_vel_acc(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size
        self.collected_data_counts_of_vel_acc = np.array(msg.data).reshape((rows, cols))

    def subscribe_collected_data_counts_of_vel_steer(self, msg):
        rows = msg.layout.dim[0].size
        cols = msg.layout.dim[1].size
        self.collected_data_counts_of_vel_steer = np.array(msg.data).reshape((rows, cols))

    def onDataCollectingArea(self, msg):
        self._data_collecting_area_polygon = msg
        self.updateNominalTargetTrajectory()

    def onGoalPose(self, msg):
        self.goal_point[0] = msg.pose.position.x
        self.goal_point[1] = msg.pose.position.y
        #self.updateNominalTargetTrajectory()

    def updateNominalTargetTrajectory(self):
        
        self.get_logger().info(" ego and goal point : " + str(self.ego_point) + " " + str(self.goal_point))
        data_collecting_area = np.array(
            [
                np.array(
                    [
                        self._data_collecting_area_polygon.polygon.points[i].x,
                        self._data_collecting_area_polygon.polygon.points[i].y,
                        self._data_collecting_area_polygon.polygon.points[i].z,
                    ]
                )
                for i in range(4)
            ]
        )

        # [1] compute an approximate rectangle
        l1 = np.sqrt(((data_collecting_area[0, :2] - data_collecting_area[1, :2]) ** 2).sum())
        l2 = np.sqrt(((data_collecting_area[1, :2] - data_collecting_area[2, :2]) ** 2).sum())
        l3 = np.sqrt(((data_collecting_area[2, :2] - data_collecting_area[3, :2]) ** 2).sum())
        l4 = np.sqrt(((data_collecting_area[3, :2] - data_collecting_area[0, :2]) ** 2).sum())
        la = (l1 + l3) / 2
        lb = (l2 + l4) / 2
        if np.abs(la - lb) < 1e-6:
            la += 0.1  # long_side_length must not be equal to short_side_length7

        rectangle_center_position = np.zeros(2)
        for i in range(4):
            rectangle_center_position[0] += data_collecting_area[i, 0] / 4.0
            rectangle_center_position[1] += data_collecting_area[i, 1] / 4.0

        vec_from_center_to_point0_data = data_collecting_area[0, :2] - rectangle_center_position
        vec_from_center_to_point1_data = data_collecting_area[1, :2] - rectangle_center_position
        unit_vec_from_center_to_point0_data = vec_from_center_to_point0_data / (
            np.sqrt((vec_from_center_to_point0_data**2).sum()) + 1e-10
        )
        unit_vec_from_center_to_point1_data = vec_from_center_to_point1_data / (
            np.sqrt((vec_from_center_to_point1_data**2).sum()) + 1e-10
        )

        # [2] compute whole trajectory
        # [2-1] generate figure eight path
        if la > lb:
            long_side_length = la
            short_side_length = lb
            vec_long_side = (
                -unit_vec_from_center_to_point0_data + unit_vec_from_center_to_point1_data
            )
        else:
            long_side_length = lb
            short_side_length = la
            vec_long_side = (
                unit_vec_from_center_to_point0_data + unit_vec_from_center_to_point1_data
            )
        unit_vec_long_side = vec_long_side / np.sqrt((vec_long_side**2).sum())
        if unit_vec_long_side[1] < 0:
            unit_vec_long_side *= -1
        yaw_offset = np.arccos(unit_vec_long_side[0])
        if yaw_offset > pi / 2:
            yaw_offset -= pi

        long_side_margin = 5
        long_side_margin = 5

        actual_long_side = max(long_side_length - long_side_margin, 1.1)
        actual_short_side = max(short_side_length - long_side_margin, 1.0)

        if self.COURSE_NAME is not None:
            (
                trajectory_position_data,
                trajectory_yaw_data,
                trajectory_curvature_data,
                self.trajectory_parts,
                self.trajectory_achievement_rates,
            ) = self.course.get_trajectory_points(actual_long_side, actual_short_side, self.ego_point, self.goal_point)

        else:
            self.trajectory_position_data = None
            self.trajectory_yaw_data = None
            self.trajectory_longitudinal_velocity_data = None
            self.trajectory_curvature_data = None

        self.get_logger().info(" trajectory_position_data : " + str(trajectory_position_data[0]))
        for i in range(len(trajectory_yaw_data)):
            if trajectory_yaw_data[i] > np.pi:
                trajectory_yaw_data[i] -= 2 * np.pi
            if trajectory_yaw_data[i] < -np.pi:
                trajectory_yaw_data[i] += 2 * np.pi

        # [2-2] translation and rotation of origin
        rot_matrix = np.array(
            [
                [np.cos(yaw_offset), -np.sin(yaw_offset)],
                [np.sin(yaw_offset), np.cos(yaw_offset)],
            ]
        )
        '''trajectory_position_data = (rot_matrix @ trajectory_position_data.T).T
        trajectory_position_data += rectangle_center_position
        trajectory_yaw_data += yaw_offset

        # [2-3] smoothing figure eight path
        window = self.get_parameter("mov_ave_window").get_parameter_value().integer_value
        self.current_window = 1 * window
        if window < len(trajectory_position_data):
            w = np.ones(window) / window
            augmented_position_data = np.vstack(
                [
                    trajectory_position_data[-window:],
                    trajectory_position_data,
                    trajectory_position_data[:window],
                ]
            )
            trajectory_position_data[:, 0] = (
                1 * np.convolve(augmented_position_data[:, 0], w, mode="same")[window:-window]
            )
            trajectory_position_data[:, 1] = (
                1 * np.convolve(augmented_position_data[:, 1], w, mode="same")[window:-window]
            )
            augmented_yaw_data = np.hstack(
                [
                    trajectory_yaw_data[-window:],
                    trajectory_yaw_data,
                    trajectory_yaw_data[:window],
                ]
            )
            smoothed_trajectory_yaw_data = trajectory_yaw_data.copy()
            for i in range(len(trajectory_yaw_data)):
                tmp_yaw = trajectory_yaw_data[i]
                tmp_data = (
                    augmented_yaw_data[window + (i - window // 2) : window + (i + window // 2)]
                    - tmp_yaw
                )
                for j in range(len(tmp_data)):
                    if tmp_data[j] > np.pi:
                        tmp_data[j] -= 2 * np.pi
                    if tmp_data[j] < -np.pi:
                        tmp_data[j] += 2 * np.pi
                tmp_data = np.convolve(tmp_data, w, mode="same")
                smoothed_trajectory_yaw_data[i] = (
                    tmp_yaw + np.convolve(tmp_data, w, mode="same")[window // 2]
                )
                if smoothed_trajectory_yaw_data[i] > np.pi:
                    smoothed_trajectory_yaw_data[i] -= 2 * np.pi
                if smoothed_trajectory_yaw_data[i] < -np.pi:
                    smoothed_trajectory_yaw_data[i] += 2 * np.pi

            trajectory_yaw_data = smoothed_trajectory_yaw_data.copy()

            if not USE_CURVATURE_RADIUS_FLAG:
                augmented_curvature_data = np.hstack(
                    [
                        trajectory_curvature_data[-window:],
                        trajectory_curvature_data,
                        trajectory_curvature_data[:window],
                    ]
                )
                trajectory_curvature_data = (
                    1 * np.convolve(augmented_curvature_data, w, mode="same")[window:-window]
                )'''
        # [2-4] nominal velocity
        target_longitudinal_velocity = (
            self.get_parameter("target_longitudinal_velocity").get_parameter_value().double_value
        )

        trajectory_longitudinal_velocity_data = target_longitudinal_velocity * np.zeros(
            len(trajectory_position_data)
        )
        self.current_target_longitudinal_velocity = 1 * target_longitudinal_velocity

        self.trajectory_position_data = trajectory_position_data.copy()
        self.trajectory_yaw_data = trajectory_yaw_data.copy()
        self.trajectory_longitudinal_velocity_data = trajectory_longitudinal_velocity_data.copy()
        self.trajectory_curvature_data = trajectory_curvature_data.copy()

        self.get_logger().info("update nominal target trajectory")

    def checkInDateCollectingArea(self, current_pos):
        data_collecting_area = np.array(
            [
                np.array(
                    [
                        self._data_collecting_area_polygon.polygon.points[i].x,
                        self._data_collecting_area_polygon.polygon.points[i].y,
                        self._data_collecting_area_polygon.polygon.points[i].z,
                    ]
                )
                for i in range(4)
            ]
        )

        A, B, C, D = (
            data_collecting_area[0][0:2],
            data_collecting_area[1][0:2],
            data_collecting_area[2][0:2],
            data_collecting_area[3][0:2],
        )
        P = current_pos[0:2]

        area_ABCD = computeTriangleArea(A, B, C) + computeTriangleArea(C, D, A)

        area_PAB = computeTriangleArea(P, A, B)
        area_PBC = computeTriangleArea(P, B, C)
        area_PCD = computeTriangleArea(P, C, D)
        area_PDA = computeTriangleArea(P, D, A)

        if area_PAB + area_PBC + area_PCD + area_PDA > area_ABCD * 1.001:
            return False
        else:
            return True

    def timer_callback_traj(self):
        if (
            self._present_kinematic_state is not None
            and self._present_acceleration is not None
            and self.trajectory_position_data is not None
        ):
            # [0] update nominal target trajectory if changing related ros2 params
            target_longitudinal_velocity = (
                self.get_parameter("target_longitudinal_velocity")
                .get_parameter_value()
                .double_value
            )

            window = self.get_parameter("mov_ave_window").get_parameter_value().integer_value

            if (
                np.abs(target_longitudinal_velocity - self.current_target_longitudinal_velocity)
                > 1e-6
                or window != self.current_window
            ):
                self.updateNominalTargetTrajectory()

            # [1] receive observation from topic
            present_position = np.array(
                [
                    self._present_kinematic_state.pose.pose.position.x,
                    self._present_kinematic_state.pose.pose.position.y,
                    self._present_kinematic_state.pose.pose.position.z,
                ]
            )
            self.ego_point = present_position[:2]

            present_orientation = np.array(
                [
                    self._present_kinematic_state.pose.pose.orientation.x,
                    self._present_kinematic_state.pose.pose.orientation.y,
                    self._present_kinematic_state.pose.pose.orientation.z,
                    self._present_kinematic_state.pose.pose.orientation.w,
                ]
            )
            present_linear_velocity = np.array(
                [
                    self._present_kinematic_state.twist.twist.linear.x,
                    self._present_kinematic_state.twist.twist.linear.y,
                    self._present_kinematic_state.twist.twist.linear.z,
                ]
            )

            if np.linalg.norm(present_orientation) < 1e-6:
                present_yaw = self.previous_yaw
            else:
                present_yaw = getYaw(present_orientation)[0]
                self.previous_yaw = present_yaw

            # [2] get whole trajectory data
            trajectory_position_data = self.trajectory_position_data.copy()
            trajectory_yaw_data = self.trajectory_yaw_data.copy()
            trajectory_longitudinal_velocity_data = (
                self.trajectory_longitudinal_velocity_data.copy()
            )
            trajectory_curvature_data = self.trajectory_curvature_data.copy()

            # [3] prepare velocity noise
            while True:
                if len(self.vel_noise_list) > len(trajectory_longitudinal_velocity_data) * 2:
                    break
                else:
                    tmp_noise_vel = (
                        np.random.rand()
                        * self.get_parameter("longitudinal_velocity_noise_amp")
                        .get_parameter_value()
                        .double_value
                    )
                    noise_min_period = (
                        self.get_parameter("longitudinal_velocity_noise_min_period")
                        .get_parameter_value()
                        .double_value
                    )
                    noise_max_period = (
                        self.get_parameter("longitudinal_velocity_noise_max_period")
                        .get_parameter_value()
                        .double_value
                    )
                    tmp_noise_period = noise_min_period + np.random.rand() * (
                        noise_max_period - noise_min_period
                    )
                    dt = self.timer_period_callback
                    noise_data_num = max(
                        4, int(tmp_noise_period / dt)
                    )  # 4 is minimum noise_data_num
                    for i in range(noise_data_num):
                        self.vel_noise_list.append(
                            tmp_noise_vel * np.sin(2.0 * np.pi * i / noise_data_num)
                        )
            self.vel_noise_list.pop(0)

            # [4] find near point index for local trajectory
            distance = np.sqrt(((trajectory_position_data - present_position[:2]) ** 2).sum(axis=1))
            index_array_near = np.argsort(distance)

            nearestIndex = None
            if (self.one_round_progress_rate is None) or (present_linear_velocity[0] < 0.1):
                # if initializing, or if re-initialize while stopping
                nearestIndex = index_array_near[0]
            else:
                for i in range(len(index_array_near)):
                    progress_rate_diff = (
                        1.0 * index_array_near[i] / len(trajectory_position_data)
                    ) - self.one_round_progress_rate
                    if progress_rate_diff > 0.5:
                        progress_rate_diff -= 1.0
                    if progress_rate_diff < -0.5:
                        progress_rate_diff += 1.0
                    near_progress_rate_threshold = 0.2
                    if np.abs(progress_rate_diff) < near_progress_rate_threshold:
                        nearestIndex = 1 * index_array_near[i]
                        break
                if nearestIndex is None:
                    nearestIndex = index_array_near[0]
            nearestIndex = index_array_near[0]
            self.one_round_progress_rate = 1.0 * nearestIndex / len(trajectory_position_data)

            # set target velocity
            present_vel = present_linear_velocity[0]
            present_acc = self._present_acceleration.accel.accel.linear.x
            target_vel = self.course.get_target_velocity(
                nearestIndex, present_vel, present_acc, self.collected_data_counts_of_vel_acc
            )

            #self.get_logger().info("target_vel: {}".format(target_vel))

            trajectory_longitudinal_velocity_data = np.array(
                [target_vel for _ in range(len(trajectory_longitudinal_velocity_data))]
            )

            # [5] modify target velocity
            # [5-1] add noise
            aug_data_length = len(trajectory_position_data) // 4
            trajectory_position_data = np.vstack(
                [trajectory_position_data, trajectory_position_data[:aug_data_length]]
            )
            trajectory_yaw_data = np.hstack(
                [trajectory_yaw_data, trajectory_yaw_data[:aug_data_length]]
            )
            trajectory_longitudinal_velocity_data = np.hstack(
                [
                    trajectory_longitudinal_velocity_data,
                    trajectory_longitudinal_velocity_data[:aug_data_length],
                ]
            )
            trajectory_longitudinal_velocity_data[nearestIndex:] += np.array(self.vel_noise_list)[
                : len(trajectory_longitudinal_velocity_data[nearestIndex:])
            ]
            trajectory_longitudinal_velocity_data_without_limit = (
                trajectory_longitudinal_velocity_data.copy()
            )

            # [5-2] apply lateral accel limit
            max_lateral_accel = (
                self.get_parameter("max_lateral_accel").get_parameter_value().double_value
            )
            if USE_CURVATURE_RADIUS_FLAG:
                lateral_acc_limit = np.sqrt(max_lateral_accel * trajectory_curvature_data)
            else:
                lateral_acc_limit = np.sqrt(max_lateral_accel / trajectory_curvature_data)
            lateral_acc_limit = np.hstack(
                [
                    lateral_acc_limit,
                    lateral_acc_limit[:aug_data_length],
                ]
            )
            if Differential_Smoothing_Flag:
                trajectory_longitudinal_velocity_data = smooth_bounding(
                    lateral_acc_limit,
                    0.9 * lateral_acc_limit,
                    trajectory_longitudinal_velocity_data,
                )
            else:
                trajectory_longitudinal_velocity_data = np.minimum(
                    trajectory_longitudinal_velocity_data, lateral_acc_limit
                )
            # [5-3] apply limit by lateral error
            velocity_limit_by_tracking_error = (
                self.get_parameter("velocity_limit_by_tracking_error")
                .get_parameter_value()
                .double_value
            )

            lateral_error_threshold = (
                self.get_parameter("lateral_error_threshold").get_parameter_value().double_value
            )

            yaw_error_threshold = (
                self.get_parameter("yaw_error_threshold").get_parameter_value().double_value
            )

            tmp_lateral_error = np.sqrt(
                ((trajectory_position_data[nearestIndex] - present_position[:2]) ** 2).sum()
            )

            tmp_yaw_error = np.abs(present_yaw - trajectory_yaw_data[nearestIndex])

            if lateral_error_threshold < tmp_lateral_error or yaw_error_threshold < tmp_yaw_error:
                if Differential_Smoothing_Flag:
                    velocity_limit_by_tracking_error_array = (
                        velocity_limit_by_tracking_error
                        * np.ones(trajectory_longitudinal_velocity_data.shape)
                    )
                    trajectory_longitudinal_velocity_data = smooth_bounding(
                        velocity_limit_by_tracking_error_array,
                        0.9 * velocity_limit_by_tracking_error_array,
                        trajectory_longitudinal_velocity_data,
                    )
                else:
                    trajectory_longitudinal_velocity_data = np.minimum(
                        trajectory_longitudinal_velocity_data, velocity_limit_by_tracking_error
                    )

            # [6] publish
            # [6-1] publish trajectory
            pub_traj_len = min(int(50 / self.traj_step), aug_data_length)
            tmp_traj = Trajectory()
            for i in range(pub_traj_len):
                tmp_traj_point = TrajectoryPoint()
                tmp_traj_point.pose.position.x = trajectory_position_data[i + nearestIndex, 0]
                tmp_traj_point.pose.position.y = trajectory_position_data[i + nearestIndex, 1]
                tmp_traj_point.pose.position.z = present_position[2]

                tmp_traj_point.pose.orientation.x = 0.0
                tmp_traj_point.pose.orientation.y = 0.0
                tmp_traj_point.pose.orientation.z = np.sin(
                    trajectory_yaw_data[i + nearestIndex] / 2
                )
                tmp_traj_point.pose.orientation.w = np.cos(
                    trajectory_yaw_data[i + nearestIndex] / 2
                )

                tmp_traj_point.longitudinal_velocity_mps = trajectory_longitudinal_velocity_data[
                    i + nearestIndex
                ]
                tmp_traj.points.append(tmp_traj_point)

            self.trajectory_for_collecting_data_pub_.publish(tmp_traj)

            # [6-2] publish marker_array
            marker_array = MarkerArray()

            # [6-2a] local trajectory
            marker_traj1 = Marker()
            marker_traj1.type = 4
            marker_traj1.id = 1
            marker_traj1.header.frame_id = "map"

            marker_traj1.action = marker_traj1.ADD

            marker_traj1.scale.x = 0.4
            marker_traj1.scale.y = 0.0
            marker_traj1.scale.z = 0.0

            marker_traj1.color.a = 1.0
            marker_traj1.color.r = 1.0
            marker_traj1.color.g = 0.0
            marker_traj1.color.b = 0.0

            marker_traj1.lifetime.nanosec = 500000000
            marker_traj1.frame_locked = True

            marker_traj1.points = []
            for i in range(len(tmp_traj.points)):
                tmp_marker_point = Point()
                tmp_marker_point.x = tmp_traj.points[i].pose.position.x
                tmp_marker_point.y = tmp_traj.points[i].pose.position.y
                tmp_marker_point.z = 0.0
                marker_traj1.points.append(tmp_marker_point)

            marker_array.markers.append(marker_traj1)

            # [6-2b] whole trajectory
            marker_traj2 = Marker()
            marker_traj2.type = 4
            marker_traj2.id = 0
            marker_traj2.header.frame_id = "map"

            marker_traj2.action = marker_traj2.ADD

            marker_traj2.scale.x = 0.2
            marker_traj2.scale.y = 0.0
            marker_traj2.scale.z = 0.0

            marker_traj2.color.a = 1.0
            marker_traj2.color.r = 0.0
            marker_traj2.color.g = 0.0
            marker_traj2.color.b = 1.0

            marker_traj2.lifetime.nanosec = 500000000
            marker_traj2.frame_locked = True

            marker_traj2.points = []
            marker_downsampling = 5
            for i in range((len(trajectory_position_data) // marker_downsampling)):
                tmp_marker_point = Point()
                tmp_marker_point.x = trajectory_position_data[i * marker_downsampling, 0]
                tmp_marker_point.y = trajectory_position_data[i * marker_downsampling, 1]
                tmp_marker_point.z = 0.0
                marker_traj2.points.append(tmp_marker_point)

            marker_array.markers.append(marker_traj2)

            # [6-2c] add arrow
            marker_arrow = Marker()
            marker_arrow.type = marker_arrow.ARROW
            marker_arrow.id = 2
            marker_arrow.header.frame_id = "map"

            marker_arrow.action = marker_arrow.ADD

            marker_arrow.scale.x = 0.5
            marker_arrow.scale.y = 2.5
            marker_arrow.scale.z = 0.0

            marker_arrow.color.a = 1.0
            marker_arrow.color.r = 1.0
            marker_arrow.color.g = 0.0
            marker_arrow.color.b = 1.0

            marker_arrow.lifetime.nanosec = 500000000
            marker_arrow.frame_locked = True

            tangent_vec = np.array([np.cos(trajectory_yaw_data[nearestIndex]), np.sin(trajectory_yaw_data[nearestIndex])])

            marker_arrow.points = []

            start_marker_point = Point()
            start_marker_point.x = tmp_traj.points[0].pose.position.x
            start_marker_point.y = tmp_traj.points[0].pose.position.y
            start_marker_point.z = 0.0
            marker_arrow.points.append(start_marker_point)

            end_marker_point = Point()
            end_marker_point.x = tmp_traj.points[0].pose.position.x + 5.0 * tangent_vec[
                0
            ] / np.linalg.norm(tangent_vec)
            end_marker_point.y = tmp_traj.points[0].pose.position.y + 5.0 * tangent_vec[
                1
            ] / np.linalg.norm(tangent_vec)
            end_marker_point.z = 0.0
            marker_arrow.points.append(end_marker_point)

            marker_array.markers.append(marker_arrow)

            self.data_collecting_trajectory_marker_array_pub_.publish(marker_array)
            # [6-3] stop request
            if not self.checkInDateCollectingArea(present_position):
                msg = Bool()
                msg.data = True
                self.pub_stop_request_.publish(msg)

            if debug_matplotlib_plot_flag:
                self.axs[0].cla()
                step_size_array = np.sqrt(
                    ((trajectory_position_data[1:] - trajectory_position_data[:-1]) ** 2).sum(
                        axis=1
                    )
                )
                distance = np.zeros(len(trajectory_position_data))
                for i in range(1, len(trajectory_position_data)):
                    distance[i] = distance[i - 1] + step_size_array[i - 1]
                distance -= distance[nearestIndex]
                time_width_array = step_size_array / (
                    trajectory_longitudinal_velocity_data[:-1] + 0.01
                )
                timestamp = np.zeros(len(trajectory_position_data))
                for i in range(1, len(trajectory_position_data)):
                    timestamp[i] = timestamp[i - 1] + time_width_array[i - 1]
                timestamp -= timestamp[nearestIndex]

                self.axs[0].plot(0, present_linear_velocity[0], "o", label="current vel")

                self.axs[0].plot(
                    timestamp[nearestIndex : nearestIndex + pub_traj_len],
                    trajectory_longitudinal_velocity_data_without_limit[
                        nearestIndex : nearestIndex + pub_traj_len
                    ],
                    "--",
                    label="target vel before applying limit",
                )
                self.axs[0].plot(
                    timestamp[nearestIndex : nearestIndex + pub_traj_len],
                    lateral_acc_limit[nearestIndex : nearestIndex + pub_traj_len],
                    "--",
                    label="lateral acc limit (always)",
                )
                self.axs[0].plot(
                    timestamp[nearestIndex : nearestIndex + pub_traj_len],
                    velocity_limit_by_tracking_error * np.ones(pub_traj_len),
                    "--",
                    label="vel limit by tracking error (only when exceeding threshold)",
                )
                self.axs[0].plot(
                    timestamp[nearestIndex : nearestIndex + pub_traj_len],
                    trajectory_longitudinal_velocity_data[
                        nearestIndex : nearestIndex + pub_traj_len
                    ],
                    label="actual target vel",
                )
                self.axs[0].set_xlim([-0.5, 10.5])
                self.axs[0].set_ylim([-0.5, 12.5])

                self.axs[0].set_xlabel("future timestamp [s]")
                self.axs[0].set_ylabel("longitudinal_velocity [m/s]")
                self.axs[0].legend(fontsize=8)

                self.fig.canvas.draw()
                plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)

    data_collecting_trajectory_publisher = DataCollectingTrajectoryPublisher()

    rclpy.spin(data_collecting_trajectory_publisher)

    data_collecting_trajectory_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
