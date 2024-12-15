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
import time

from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import ControlModeReport
from autoware_vehicle_msgs.msg import GearCommand
from autoware_vehicle_msgs.msg import GearReport
from autoware_vehicle_msgs.srv import ControlModeCommand
import rclpy
from tier4_vehicle_msgs.msg import ActuationCommandStamped

'''from j6_interface_msgs.msg import T0Debug'''


def call_control_mode_request(node, mode):
    request = ControlModeCommand.Request()
    request.mode = mode

    future = node.client_control_mode.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        print(f"Response from control mode service: {future.result()}")
    else:
        print("Control mode service call failed.")
        sys.exit()


def change_mode(node, mode):
    print(f"Change mode to {mode}")
    if mode == "autonomous":
        request = ControlModeCommand.Request.AUTONOMOUS
        report = ControlModeReport.AUTONOMOUS
    elif mode == "autonomous_velocity_only":
        request = ControlModeCommand.Request.AUTONOMOUS_VELOCITY_ONLY
        report = ControlModeReport.AUTONOMOUS_VELOCITY_ONLY
    elif mode == "autonomous_steering_only":
        request = ControlModeCommand.Request.AUTONOMOUS_STEER_ONLY
        report = ControlModeReport.AUTONOMOUS_STEER_ONLY
    elif mode == "manual":
        request = ControlModeCommand.Request.MANUAL
        report = ControlModeReport.MANUAL
    else:
        print(f"Invalid mode: {mode}")
        sys.exit(1)

    call_control_mode_request(node, request)

    while rclpy.ok():
        rclpy.spin_once(node)
        if node.current_control_mode == report:
            break


def change_gear(node, target_gear):
    print(f"Change gear to {target_gear}")
    if target_gear == "neutral":
        command = GearCommand.NEUTRAL
        report = GearReport.NEUTRAL
    elif target_gear == "drive":
        command = GearCommand.DRIVE
        report = GearReport.DRIVE
    elif target_gear == "reverse":
        command = GearCommand.REVERSE
        report = GearReport.REVERSE
    else:
        print(f"Invalid gear: {target_gear}")
        sys.exit(1)

    gear_cmd_msg = GearCommand()
    gear_cmd_msg.stamp = node.get_clock().now().to_msg()
    gear_cmd_msg.command = command
    node.pub_gear_cmd.publish(gear_cmd_msg)

    while rclpy.ok():
        rclpy.spin_once(node)
        if node.current_gear == report:
            print(f"Current gear is {target_gear}")
            break


'''def change_sub_brake_mode(node, activate):
    if activate:
        print("Turn on forced sub braking mode.")
    else:
        print("Turn off forced sub braking mode.")

    t0_debug_command_msg = T0Debug()
    t0_debug_command_msg.stamp = node.get_clock().now().to_msg()
    t0_debug_command_msg.forced_sub_braking = activate
    t0_debug_command_msg.ebs_xpb_fault_injection = activate
    node.pub_t0_debug_cmd.publish(t0_debug_command_msg)
    time.sleep(0.5)'''


def accelerate(
    node, target_acceleration, target_velocity, mode, target_jerk=None, use_sub_brake=False
):
    print(f"Accelerate with {target_acceleration} m/s^2.")
    '''if use_sub_brake:
        change_sub_brake_mode(node, True)'''

    if target_jerk == None:
        acceleration_cmd = target_acceleration
    else:
        acceleration_cmd = 0.0

    control_cmd_msg = Control()
    condition = (
        lambda: acceleration_cmd < target_acceleration - 1e-3
        if mode == "drive"
        else acceleration_cmd > target_acceleration + 1e-3
    )
    while condition():
        acceleration_cmd += target_jerk / 10.0
        control_cmd_msg.stamp = node.get_clock().now().to_msg()
        control_cmd_msg.longitudinal.acceleration = acceleration_cmd
        control_cmd_msg.lateral.steering_tire_angle = 0.0
        node.pub_control_cmd.publish(control_cmd_msg)
        time.sleep(0.1)

    control_cmd_msg.stamp = node.get_clock().now().to_msg()
    control_cmd_msg.longitudinal.acceleration = target_acceleration
    control_cmd_msg.lateral.steering_tire_angle = 0.0
    node.pub_control_cmd.publish(control_cmd_msg)

    node.target_acceleration = target_acceleration
    node.control_cmd_timer.reset()

    while rclpy.ok():
        rclpy.spin_once(node)
        if (mode == "drive" and node.current_velocity * 3.6 >= target_velocity) or (
            mode == "brake" and node.current_velocity * 3.6 <= target_velocity
        ):
            print(f"Reached {target_velocity} km/h.")
            node.control_cmd_timer.cancel()
            control_cmd_msg.stamp = node.get_clock().now().to_msg()
            control_cmd_msg.longitudinal.acceleration = 0.0 if mode == "drive" else -2.5
            control_cmd_msg.lateral.steering_tire_angle = 0.0
            node.pub_control_cmd.publish(control_cmd_msg)
            break
    time.sleep(1)

    if use_sub_brake:
        change_sub_brake_mode(node, False)


def steer(node, target_steering_tire_angle):
    print(f"Steer to {target_steering_tire_angle} rad.")
    control_cmd_msg = Control()
    control_cmd_msg.stamp = node.get_clock().now().to_msg()
    control_cmd_msg.lateral.steering_tire_angle = target_steering_tire_angle
    node.pub_control_cmd.publish(control_cmd_msg)


def actuate(node, mode, target_command, target_velocity, use_sub_brake=False, break_time = 120.0):
    print(f"Actuate with {mode} command: {target_command}.")
    start_time = time.time()
    if use_sub_brake:
        pass
        #change_sub_brake_mode(node, True)

    actuation_cmd_msg = ActuationCommandStamped()
    if mode == "accel":
        actuation_cmd_msg.actuation.accel_cmd = target_command
    elif mode == "brake" or mode == "sub_brake":
        actuation_cmd_msg.actuation.brake_cmd = target_command
    else:
        print(f"Invalid mode: {mode}")
        sys.exit(1)

    actuation_cmd_msg.header.stamp = node.get_clock().now().to_msg()
    if mode == "sub_brake":
        node.pub_sub_actuation_cmd.publish(actuation_cmd_msg)
    else:
        node.pub_actuation_cmd.publish(actuation_cmd_msg)

    while rclpy.ok():
        rclpy.spin_once(node)
        if (
            (mode == "accel" and node.current_velocity * 3.6 >= target_velocity)
            or (mode == "brake" and node.current_velocity * 3.6 <= target_velocity)
            or (mode == "sub_brake" and node.current_velocity * 3.6 <= target_velocity)
        ):
            print(f"Reached {target_velocity} km/h.")
            actuation_cmd_msg.header.stamp = node.get_clock().now().to_msg()
            actuation_cmd_msg.actuation.accel_cmd = 0.0
            actuation_cmd_msg.actuation.brake_cmd = 0.0
            if mode == "sub_brake":
                node.pub_sub_actuation_cmd.publish(actuation_cmd_msg)
            else:
                node.pub_actuation_cmd.publish(actuation_cmd_msg)
            break
        
        if time.time() - start_time > break_time:
            print("break : " + str(break_time) + " has passed.")
            break

    time.sleep(1)

    if use_sub_brake:
        pass
        #change_sub_brake_mode(node, False)


def reset_commands(node):
    control_cmd_msg = Control()
    control_cmd_msg.stamp = node.get_clock().now().to_msg()
    control_cmd_msg.longitudinal.acceleration = 0.0
    control_cmd_msg.lateral.steering_tire_angle = 0.0
    node.pub_control_cmd.publish(control_cmd_msg)
    print("Reset control command.")

    actuation_cmd_msg = ActuationCommandStamped()
    actuation_cmd_msg.header.stamp = node.get_clock().now().to_msg()
    actuation_cmd_msg.actuation.accel_cmd = 0.0
    actuation_cmd_msg.actuation.brake_cmd = 0.0
    node.pub_actuation_cmd.publish(actuation_cmd_msg)
    print("Reset actuation command.")

    gear_cmd_msg = GearCommand()
    gear_cmd_msg.stamp = node.get_clock().now().to_msg()
    gear_cmd_msg.command = GearCommand.NEUTRAL
    node.pub_gear_cmd.publish(gear_cmd_msg)
    print("Reset gear command.")
