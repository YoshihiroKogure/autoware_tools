#!/usr/bin/env python3

# Copyright 2024 Tier IV, Inc. All rights reserved.
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

import sys

from autoware_vehicle_msgs.msg import ControlModeReport
import numpy as np


def get_velocity_for_calculating_acceleration(
    map_axis_velocity_kph, min_velocity_for_calc_accel_kph
):
    axis_velocity_step = map_axis_velocity_kph[-1] - map_axis_velocity_kph[-2]
    map_axis_velocity_kph.append(map_axis_velocity_kph[-1] + axis_velocity_step)

    velocity_for_calculating_acceleration = []
    for i in range(len(map_axis_velocity_kph) - 1):
        velocity = np.average([map_axis_velocity_kph[i], map_axis_velocity_kph[i + 1]])
        if i == 0:
            velocity = max(velocity, min_velocity_for_calc_accel_kph)
        velocity_for_calculating_acceleration.append(velocity * 1000.0 / 60.0 / 60.0)

    return velocity_for_calculating_acceleration


def get_stamps_for_calculating_acceleration(
    velocity_for_calculating_acceleration,
    topics,
    stamp_target_section_start,
    stamp_target_section_end,
    velocity_margin_kph,
    nearest_velocity_threshold_kph,
):
    stamps_for_calculating_acceleration = []
    for velocity in velocity_for_calculating_acceleration:
        stamp = get_stamp_when_target_velocity(
            topics,
            velocity,
            stamp_target_section_start,
            stamp_target_section_end,
            velocity_margin_kph,
            nearest_velocity_threshold_kph,
        )
        print(f"velocity: {velocity:.2f}, stamp: {stamp}")
        stamps_for_calculating_acceleration.append(stamp)

    return stamps_for_calculating_acceleration


def get_autonomous_mode_stamps(topics):
    stamp_autonomous_start = 0
    stamp_autonomous_end = int(1e20)
    is_autonomous = False

    for control_mode_msg in topics["/vehicle/status/control_mode"]:
        if control_mode_msg.mode != ControlModeReport.MANUAL and not is_autonomous:
            if stamp_autonomous_start != 0:
                print("===== error!: driving auto twice in one rosbag =====")
                sys.exit()
            stamp_autonomous_start = round(
                control_mode_msg.stamp.sec + control_mode_msg.stamp.nanosec * 1e-9, 3
            )
            is_autonomous = True
        if control_mode_msg.mode == ControlModeReport.MANUAL and is_autonomous:
            stamp_autonomous_end = round(
                control_mode_msg.stamp.sec + control_mode_msg.stamp.nanosec * 1e-9, 3
            )
            is_autonomous = False

    print(
        f"stamp_autonomous_start: {stamp_autonomous_start}, stamp_autonomous_end: {stamp_autonomous_end}"
    )
    return (stamp_autonomous_start, stamp_autonomous_end)


def get_actuation_cmd(topics, actuation_type):
    buffer = []
    if actuation_type == "accel" or actuation_type == "brake":
        actuation_cmd_msgs = topics["/control/command/actuation_cmd"]
    elif actuation_type == "sub_brake":
        actuation_cmd_msgs = topics["/control/command/sub_actuation_cmd"]
    else:
        print(f"Invalid actuation type: {actuation_type}")
        sys.exit(1)

    for actuation_cmd_msg in actuation_cmd_msgs:
        if actuation_type == "accel" and actuation_cmd_msg.actuation.accel_cmd != 0:
            buffer.append(actuation_cmd_msg.actuation.accel_cmd)
        if actuation_type == "brake" and actuation_cmd_msg.actuation.brake_cmd != 0:
            buffer.append(actuation_cmd_msg.actuation.brake_cmd)
        if actuation_type == "sub_brake" and actuation_cmd_msg.actuation.brake_cmd != 0:
            buffer.append(actuation_cmd_msg.actuation.brake_cmd)

    if not buffer:
        return 0
    else:
        return round(np.average(buffer), 1)


def get_actuation_stamps(
    topics, actuation_type, target_actuation_cmd, stamp_auto_driving_start, stamp_auto_driving_end
):
    stamp_actuation_start = 0
    stamp_actuation_end = int(1e20)
    is_start = False

    if actuation_type == "accel" or actuation_type == "brake":
        msgs = topics["/control/command/actuation_cmd"]
    elif actuation_type == "sub_brake":
        msgs = topics["/control/command/sub_actuation_cmd"]
    else:
        print(f"Invalid actuation type: {actuation_type}")
        sys.exit(1)

    for actuation_cmd_msg in topics["/control/command/actuation_cmd"]:
        stamp = actuation_cmd_msg.header.stamp.sec + actuation_cmd_msg.header.stamp.nanosec * 1e-9
        if stamp < stamp_auto_driving_start or stamp_auto_driving_end < stamp:
            continue

        if actuation_type == "accel":
            actuation_cmd = actuation_cmd_msg.actuation.accel_cmd
        elif actuation_type == "brake":
            actuation_cmd = actuation_cmd_msg.actuation.brake_cmd

        if (
            target_actuation_cmd * 0.9 < actuation_cmd
            and actuation_cmd < target_actuation_cmd * 1.1
        ):
            if is_start == False:
                stamp_actuation_start = stamp
                is_start = True

            stamp_actuation_end = stamp

    return (stamp_actuation_start, stamp_actuation_end)


def get_target_acceleration(topics, mode):
    buffer = []
    for control_cmd_msg in topics["/control/command/control_cmd"]:
        if mode == "accel" and control_cmd_msg.longitudinal.acceleration > 0:
            buffer.append(control_cmd_msg.longitudinal.acceleration)
        elif (
            mode == "brake" or mode == "sub_brake"
        ) and control_cmd_msg.longitudinal.acceleration <= 0:
            buffer.append(control_cmd_msg.longitudinal.acceleration)

    if not buffer:
        print("Buffer is empty for calculating target acceleration. return 0.0.")
        return 0
    else:
        target_acceleration = round(np.average(buffer), 2)
        print(f"Target acceleration: {target_acceleration}")
        return target_acceleration


def get_acceleration_stamps(
    topics, mode, target_acceleration, stamp_autonomous_start, stamp_autonomous_end
):
    stamp_acceleration_start = 0.0
    stamp_acceleration_end = int(1e20)
    is_start = False

    for control_cmd_msg in topics["/control/command/control_cmd"]:
        stamp = round(control_cmd_msg.stamp.sec + control_cmd_msg.stamp.nanosec * 1e-9, 3)
        if stamp < stamp_autonomous_start or stamp_autonomous_end < stamp:
            continue

        acceleration_cmd = control_cmd_msg.longitudinal.acceleration
        condition = (
            lambda: target_acceleration * 0.9 < acceleration_cmd
            and acceleration_cmd < target_acceleration * 1.1
            if mode == "accel"
            else target_acceleration * 0.9 > acceleration_cmd
            and acceleration_cmd > target_acceleration * 1.1
        )
        if condition():
            if is_start == False:
                stamp_acceleration_start = stamp
                is_start = True

            stamp_acceleration_end = stamp

    print(
        f"stamp_acceleration_start: {stamp_acceleration_start}, stamp_acceleration_end: {stamp_acceleration_end}"
    )
    return (stamp_acceleration_start, stamp_acceleration_end)


def get_stamp_when_target_velocity(
    topics,
    target_velocity,
    stamp_target_section_start,
    stamp_target_section_end,
    velocity_margin_kph,
    nearest_velocity_threshold_kph,
):
    nearest_diff_velocity = 10000
    nearest_stamp = 0.0
    stamps_in_velocity_margin = []

    for velocity_status_msg in topics["/vehicle/status/velocity_status"]:
        stamp = round(
            velocity_status_msg.header.stamp.sec + velocity_status_msg.header.stamp.nanosec * 1e-9,
            3,
        )
        if stamp < stamp_target_section_start or stamp_target_section_end < stamp:
            continue

        diff_velocity = velocity_status_msg.longitudinal_velocity - target_velocity
        if abs(diff_velocity) < abs(nearest_diff_velocity):
            nearest_diff_velocity = diff_velocity
            nearest_stamp = stamp

        if abs(diff_velocity) < velocity_margin_kph / 3.6:
            stamps_in_velocity_margin.append(stamp)

    if stamps_in_velocity_margin:
        return np.average(stamps_in_velocity_margin)
    if abs(nearest_diff_velocity) < nearest_velocity_threshold_kph / 3.6:
        return nearest_stamp

    print(f"No data corresponding to target velocity: {target_velocity * 3.6:.1f} km/h")
    return 0


def collect_offset_imu_acceleration(
    imu_acceleration_time, imu_acceleration_mps2, velocity_time, velocity_mps
):
    is_stop = False

    stamp_stop_start = 0
    stamp_stop_end = int(1e20)
    for v, t in zip(velocity_mps, velocity_time):
        if v < 1e-3 and not is_stop:
            stamp_stop_start = t
            is_stop = True
        if v > 1e-3 and is_stop:
            stamp_stop_end = t
            break
    print(f"stamp_stop_start: {stamp_stop_start}, stamp_stop_end: {stamp_stop_end}")

    buffer = []
    for a, t in zip(imu_acceleration_mps2, imu_acceleration_time):
        if stamp_stop_start + 1.0 <= t and t <= stamp_stop_end - 1.0:
            buffer.append(a)
    if buffer:
        offset = round(np.average(buffer), 4)
    else:
        print("Buffer for calculating IMU offset is empty. Offset is 0.0.")
        offset = 0.0
    print(f"IMU acceleration offset: {offset}")

    return list(np.array(imu_acceleration_mps2) - offset)
