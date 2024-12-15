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

from datetime import datetime
import os
import subprocess
import sys
import time

from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import ControlModeReport
from autoware_vehicle_msgs.msg import GearCommand
from autoware_vehicle_msgs.msg import GearReport
from autoware_vehicle_msgs.msg import VelocityReport
from autoware_vehicle_msgs.srv import ControlModeCommand
import lib.command
import lib.cui
import lib.rosbag
import lib.system
import rclpy
from rclpy.node import Node
from tier4_vehicle_msgs.msg import ActuationCommandStamped

#from j6_interface_msgs.msg import T0Debug

COUNTDOWN_TIME = 3  # [sec]
TARGET_VELOCITY = 20.0  # [km/h]
#TARGET_ACCELERATION_FOR_DRIVE = 1.5  # [m/s^2]
#TARGET_ACCELERATION_FOR_BRAKE = -1.5  # [m/s^2]
TARGET_ACTUATION_FOR_ACCEL = 0.3
TARGET_ACTUATION_FOR_BRAKE = 0.5 
TARGET_JERK_FOR_DRIVE = 1.5  # [m/s^3]
TARGET_JERK_FOR_BRAKE = -1.5  # [m/s^3]
TARGET_WEIGHT = 440  # [kg]

#MIN_ACCEL = -6.0
#MAX_ACCEL = 2.0
#MIN_ACCEL_SUB_BRAKE = -2.0

MAX_ACCEL_PEDAL = 0.5
MIN_BRAKE_PEDAL = 0.8

TOPIC_LIST_FOR_VALIDATION = [
    "/vehicle/status/velocity_status",
    "/control/command/actuation_cmd",
    "/sensing/imu/imu_data",
    "/vehicle/status/control_mode",
]

NODE_LIST_FOR_VALIDATION = [
    #"/j6/j6_can_interface",
    #"/j6/j6_interface",
    #"/main/raw_vehicle_cmd_converter",
    #"/sub/raw_vehicle_cmd_converter",
    "/raw_vehicle_cmd_converter"
    #"/velocity_status_odometry_converter",
    #"/imu/tamagawa/tag_can_driver",
]


class MapAccuracyTester(Node):
    def __init__(self):
        super().__init__("map_accuracy_tester")
        self.client_control_mode = self.create_client(
            ControlModeCommand, "/control/control_mode_request"
        )

        while not self.client_control_mode.wait_for_service(timeout_sec=1.0):
            print("Waiting for the control mode service to become available...")

        self.pub_control_cmd = self.create_publisher(Control, "/control/command/control_cmd", 1)
        self.pub_gear_cmd = self.create_publisher(GearCommand, "/control/command/gear_cmd", 1)
        #self.pub_t0_debug_cmd = self.create_publisher(
            #T0Debug, "/j6/can/command/t0_debug_command", 1
        #)
        self.sub_velocity_status = self.create_subscription(
            VelocityReport, "/vehicle/status/velocity_status", self.on_velocity_status, 1
        )
        self.sub_control_mode = self.create_subscription(
            ControlModeReport, "/vehicle/status/control_mode", self.on_control_mode, 1
        )
        self.sub_gear_status = self.create_subscription(
            GearReport, "/vehicle/status/gear_status", self.on_gear_status, 1
        )

        self.control_cmd_timer = self.create_timer(0.03, self.control_cmd_timer_callback)
        self.control_cmd_timer.cancel()
        self.target_acceleration = 0.0

        self.current_velocity = 0.0
        self.current_control_mode = ControlModeReport.MANUAL
        self.current_gear = GearReport.NONE

        # For commands reset
        self.pub_actuation_cmd = self.create_publisher(
            ActuationCommandStamped, "/control/command/actuation_cmd", 1
        )

        self.vehicle_id = "1"#os.getenv("VEHICLE_ID")
        if not self.vehicle_id:
            print("VEHICLE_ID is not set.")
            sys.exit(1)

    def on_velocity_status(self, msg):
        self.current_velocity = msg.longitudinal_velocity

    def on_control_mode(self, msg):
        self.current_control_mode = msg.mode

    def on_gear_status(self, msg):
        self.current_gear = msg.report

    def control_cmd_timer_callback(self):
        control_cmd_msg = Control()
        control_cmd_msg.stamp = self.get_clock().now().to_msg()
        control_cmd_msg.longitudinal.acceleration = self.target_acceleration
        self.pub_control_cmd.publish(control_cmd_msg)

    def run(self):
        print("===== Start actuate tester =====")
        lib.system.check_service_active("autoware.service")
        lib.system.check_node_active(NODE_LIST_FOR_VALIDATION)

        lib.cui.ready_check(
            f"Ready to run the script?\nIs the vehicle weight adjusted to {TARGET_WEIGHT}kg?"
        )

        print("===== Reset commands =====")
        lib.command.reset_commands(self)

        print("===== Start checking accel map =====")
        lib.cui.do_check("Do you want to accel pedal data?", lambda: self.check("accel"))

        print("===== Start checking brake map =====")
        lib.cui.do_check("Do you want to brake pedal data?", lambda: self.check("brake"))

        print("===== Successfully finished! =====")

    def check(self, mode):
        is_finished = False
        while not is_finished:
            print("===== Input target accel pedal input =====")
            min_actuation, max_actuation = self.get_min_max_acceleration(mode)
            target_actuation = lib.cui.input_target_value(
                mode + " actuation", min_actuation, max_actuation, ""
            )

            

            if mode == "accel":

                print("===== Record rosbag =====")
                filename = self.get_rosbag_name(mode, target_actuation)
                process = lib.rosbag.record_ros2_bag(filename, lib.rosbag.TOPIC_LIST)
                lib.cui.countdown(COUNTDOWN_TIME)
                print(f"record rosbag: {filename}")

                print("===== Enter velocity only autonomous mode =====")
                #lib.cui.ready_check("Ready to enter velocity only autonomous mode?")
                lib.cui.countdown(COUNTDOWN_TIME)
                #lib.command.change_mode(self, "autonomous_velocity_only")

                print(
                    f"===== Drive to {TARGET_VELOCITY} km/h with accel pedal actuation {target_actuation} ====="
                )
                lib.command.change_gear(self, "drive")
                lib.cui.ready_check("Ready to drive?")
                lib.cui.countdown(COUNTDOWN_TIME)
                lib.command.actuate(self, mode, target_actuation, TARGET_VELOCITY, break_time=30.0)
                print("===== End rosbag record =====")
                process.terminate()
                lib.command.actuate(
                    self, "brake", TARGET_ACTUATION_FOR_BRAKE, 1e-3
                )
            elif mode == "brake":
                print(
                    f"===== Drive to {TARGET_VELOCITY} km/h and brake pedal actuation with {target_actuation} ====="
                )
                lib.command.change_gear(self, "drive")
                lib.cui.ready_check("Ready to drive?")
                lib.cui.countdown(COUNTDOWN_TIME)
                lib.command.actuate(
                    self,
                    "accel",
                    TARGET_ACTUATION_FOR_ACCEL,
                    TARGET_VELOCITY,
                )
                filename = self.get_rosbag_name(mode, target_actuation)
                process = lib.rosbag.record_ros2_bag(filename, lib.rosbag.TOPIC_LIST)
                print("===== Record rosbag =====")
                print(f"record rosbag: {filename}")

                #print("===== Enter velocity only autonomous mode =====")
                #lib.cui.ready_check("Ready to enter velocity only autonomous mode?")
                #lib.cui.countdown(COUNTDOWN_TIME)
                #lib.command.change_mode(self, "autonomous_velocity_only")
                lib.command.actuate(self, mode, target_actuation, 1e-3, break_time=60.0)
                print("===== End rosbag record =====")
                process.terminate()
            else:
                print(f"Invalid mode: {mode}")
                sys.exit(1)

            process.wait()

            print("===== Validate rosbag =====")
            is_rosbag_valid = lib.rosbag.validate(filename, TOPIC_LIST_FOR_VALIDATION)
            if not is_rosbag_valid:
                print(f"Rosag validation error: {filename}")
                sys.exit(1)

            is_finished = lib.cui.finish_check(f"Will you continue to check {mode} map?")

        print(f"===== Successfully {mode} map checking finished! =====")

    def get_min_max_acceleration(self, mode):
        if mode == "accel":
            return 0.0, MAX_ACCEL_PEDAL
        if mode == "brake":
            return 0.0, MIN_BRAKE_PEDAL

    def get_rosbag_name(self, mode, target_acceleration):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "_".join(
            [
                self.vehicle_id,
                str(TARGET_WEIGHT) + "kg",
                "acceleration_accuracy",
                mode,
                str(target_acceleration),
                current_time,
            ]
        )
        return filename


def main(args=None):
    rclpy.init(args=args)

    tester = MapAccuracyTester()
    tester.run()

    tester.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
