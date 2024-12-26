#!/usr/bin/env python3

# Copyright 2024 TIER IV, Inc. All rights reserved.
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

import argparse
import glob
from logging import getLogger
import os
import pprint
import sys

import lib.map
import lib.rosbag
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from tabulate import tabulate

# Script parameters
CURRENT_VELOCITY_TOPIC_NAME = "/vehicle/status/velocity_status"
CONTROL_COMMAND_TOPIC_NAME = "/control/command/control_cmd"
ACTUATION_COMMAND_TOPIC_NAME = "/control/command/actuation_cmd"
SUB_ACTUATION_COMMAND_TOPIC_NAME = "/control/command/sub_actuation_cmd"
CONTROL_MODE_TOPIE_NAME = "/vehicle/status/control_mode"
IMU_TOPIC_NAME = "/sensing/imu/imu_data"
WHEEL_SPEED_TOPIC_NAME = "/vehicle/status/wheel_speed_status"
TOPIC_NAMES = [
    CURRENT_VELOCITY_TOPIC_NAME,
    CONTROL_COMMAND_TOPIC_NAME,
    ACTUATION_COMMAND_TOPIC_NAME,
    SUB_ACTUATION_COMMAND_TOPIC_NAME,
    CONTROL_MODE_TOPIE_NAME,
    IMU_TOPIC_NAME,
    WHEEL_SPEED_TOPIC_NAME,
]
MAP_AXIS_VELOCITY_KPH = [0, 5, 10, 15, 20, 25, 30, 35]
MIN_VELOCITY_FOR_CALC_ACCEL_KPH = 3
VELOCITY_MARGIN_KPH = 0.1
NEAREST_VELOCITY_THRESHOLD_KPH = 0.3
CRITERIA_RANGE = 0.1
CRITERIA_MIN_ACCEL = 0.1


def average_within_time_range(value_array, time_array, start_time, end_time):
    filtered_value_array = []
    for value, time in zip(value_array, time_array):
        if start_time <= time <= end_time:
            filtered_value_array.append(value)

    if filtered_value_array:
        average_value = sum(filtered_value_array) / len(filtered_value_array)
        return average_value
    else:
        print("No values found within the specified time range.")
        sys.exit(1)


class MapAccuracyChecker:
    def __init__(self, data_path, report_path, vehicle_name):
        self.logger = getLogger(__name__)

        pprint.pprint(data_path + "/accel/*/*.db3")
        pprint.pprint(glob.glob(data_path + "/accel/*/*.db3"))
        self.accel_data_paths = glob.glob(data_path + "/accel/*/*.db3")
        if not self.accel_data_paths:
            self.logger.info("Accel data is empty.")
        self.brake_data_paths = glob.glob(data_path + "/brake/*/*.db3")
        if not self.brake_data_paths:
            self.logger.info("Brake data is empty.")
        self.sub_brake_data_paths = glob.glob(data_path + "/sub_brake/*/*.db3")
        if not self.sub_brake_data_paths:
            self.logger.info("Sub brake data is empty.")

        self.report_path = report_path
        self.vehicle_name = vehicle_name
        self.output_dir_use = os.path.join(self.report_path, "use")
        self.output_dir_not_use = os.path.join(self.report_path, "not_use")
        if not os.path.exists(self.output_dir_use):
            os.makedirs(self.output_dir_use)
        if not os.path.exists(self.output_dir_not_use):
            os.makedirs(self.output_dir_not_use)

        self.velocity_for_calculating_acceleration = (
            lib.map.get_velocity_for_calculating_acceleration(
                MAP_AXIS_VELOCITY_KPH, MIN_VELOCITY_FOR_CALC_ACCEL_KPH
            )
        )
        print(self.velocity_for_calculating_acceleration)

        self.acceleration_results = {"accel": {}, "brake": {}, "sub_brake": {}}
        self.acceleration_results_each_file = {"accel": {}, "brake": {}, "sub_brake": {}}

    def run(self):
        for data_path in self.accel_data_paths:
            self.logger.info(f"Start: {data_path}")
            topics = lib.rosbag.get_topics(data_path, TOPIC_NAMES)
            title = os.path.splitext(os.path.basename(data_path))[0]
            self.plot(topics, title, "accel")

        for data_path in self.brake_data_paths:
            self.logger.info(f"Start: {data_path}")
            topics = lib.rosbag.get_topics(data_path, TOPIC_NAMES)
            title = os.path.splitext(os.path.basename(data_path))[0]
            self.plot(topics, title, "brake")

        '''for data_path in self.sub_brake_data_paths:
            self.logger.info(f"Start: {data_path}")
            topics = lib.rosbag.get_topics(data_path, TOPIC_NAMES)
            title = os.path.splitext(os.path.basename(data_path))[0]
            self.plot(topics, title, "sub_brake")'''

        self.report()

    def plot(self, topics, title, mode):
        # Ready plot data
        velocity_time = []
        velocity_mps = []
        acceleration_command_time = []
        acceleration_command_mps2 = []
        calculated_acceleration_time = []
        calculated_acceleration_mps2 = []
        imu_acceleration_time = []
        imu_acceleration_mps2 = []
        actuation_command_time = []
        actuation_command_value = []
        wheel_speed_time = []
        wheel_speed_front_left_mps = []
        wheel_speed_front_right_mps = []
        wheel_speed_rear_left_mps = []
        wheel_speed_rear_right_mps = []

        for velocity_status_msg in topics[CURRENT_VELOCITY_TOPIC_NAME]:
            velocity_time.append(
                round(
                    velocity_status_msg.header.stamp.sec
                    + velocity_status_msg.header.stamp.nanosec * 1e-9,
                    3,
                )
            )
            velocity_mps.append(velocity_status_msg.longitudinal_velocity)

        print(velocity_mps)
        for control_cmd_msg in topics[CONTROL_COMMAND_TOPIC_NAME]:
            acceleration_command_time.append(
                round(control_cmd_msg.stamp.sec + control_cmd_msg.stamp.nanosec * 1e-9, 3)
            )
            acceleration_command_mps2.append(control_cmd_msg.longitudinal.acceleration)

        (
            acceleration_result,
            calculated_acceleration_time,
            calculated_acceleration_mps2,
            plot_info,
        ) = self.calculate_acceleration_from_velocity(topics, mode)

        for imu_msg in topics[IMU_TOPIC_NAME]:
            imu_acceleration_time.append(
                round(imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9, 3)
            )
            imu_acceleration_mps2.append(imu_msg.linear_acceleration.x)

        # IMU linear acceleration offset collection
        imu_acceleration_offset_collected_mps2 = lib.map.collect_offset_imu_acceleration(
            imu_acceleration_time, imu_acceleration_mps2, velocity_time, velocity_mps
        )

        if mode == "accel" or mode == "brake":
            msgs = topics[ACTUATION_COMMAND_TOPIC_NAME]
        elif mode == "sub_brake":
            #msgs = topics[SUB_ACTUATION_COMMAND_TOPIC_NAME]
            pass
        else:
            print(f"Invalid actuation type: {mode}")
            sys.exit(1)
        for actuation_cmd_msg in msgs:
            actuation_command_time.append(
                round(
                    actuation_cmd_msg.header.stamp.sec
                    + actuation_cmd_msg.header.stamp.nanosec * 1e-9,
                    3,
                )
            )
            if mode == "accel":
                actuation_command_value.append(actuation_cmd_msg.actuation.accel_cmd)
            else:
                actuation_command_value.append(actuation_cmd_msg.actuation.brake_cmd)

        '''for wheel_speed_msg in topics[WHEEL_SPEED_TOPIC_NAME]:
            wheel_speed_time.append(
                round(wheel_speed_msg.stamp.sec + wheel_speed_msg.stamp.nanosec * 1e-9, 3)
            )
            wheel_speed_front_left_mps.append(wheel_speed_msg.front_left_wheel_speed)
            wheel_speed_front_right_mps.append(wheel_speed_msg.front_right_wheel_speed)
            wheel_speed_rear_left_mps.append(wheel_speed_msg.rear_left_wheel_speed)
            wheel_speed_rear_right_mps.append(wheel_speed_msg.rear_right_wheel_speed)'''

        # Plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8.0, 12.0), sharex=True)
        plt.subplots_adjust(left=0.1, bottom=0.2)

        # ax1 to plot velocity
        ax1.plot(np.array(velocity_time), velocity_mps, label="velocity")
        ax1.scatter(
            plot_info["stamps_for_calculating_acceleration_non_zero"],
            plot_info["velocity_for_calculating_acceleration_non_zero"],
            label="accel calc points",
            s=10,
            c="red",
        )
        '''ax1.axvline(
            plot_info["stamp_acceleration_start"],
            color="g",
            linestyle="--",
            label="Search section start",
        )
        ax1.axvline(
            plot_info["stamp_acceleration_end"],
            color="g",
            linestyle="--",
            label="Search section end",
        )'''
        ax1.set_title(title, fontsize=10)
        ax1.set_ylabel("Velocity [m/s]")
        ax1.legend()

        # ax2 to plot acceleration
        ax2.plot(
            np.array(acceleration_command_time),
            acceleration_command_mps2,
            label="acceleration command",
            zorder=2,
        )
        ax2.plot(
            np.array(imu_acceleration_time),
            imu_acceleration_offset_collected_mps2,
            label="imu acceleration",
            zorder=1,
        )
        ax2.scatter(
            np.array(calculated_acceleration_time),
            calculated_acceleration_mps2,
            label="calculated acceleration",
            s=10,
            c="red",
            zorder=3,
        )
        ax2.axhline(
            plot_info["target_acceleration"] * 1.1, color="r", linestyle="--", label="criteria(max)"
        )
        ax2.axhline(
            plot_info["target_acceleration"] * 0.9, color="r", linestyle="--", label="criteria(min)"
        )
        ax2.set_ylim(plot_info["target_acceleration"] - 1.0, plot_info["target_acceleration"] + 1.0)
        ax2.set_ylabel("Acceleration [m/s^2]")
        ax2.legend()

        # ax3 to plot actuation
        ax3.plot(np.array(actuation_command_time), actuation_command_value, label="actuation command")
        ax3.set_ylabel("Actuation command [-]")
        ax3.legend()

        # ax4 to plot wheel speed
        '''ax4.plot(wheel_speed_time, wheel_speed_front_left_mps, label="front left")
        ax4.plot(wheel_speed_time, wheel_speed_front_right_mps, label="front right")
        ax4.plot(wheel_speed_time, wheel_speed_rear_left_mps, label="rear left")
        ax4.plot(wheel_speed_time, wheel_speed_rear_right_mps, label="rear right")'''
        ax4.set_xlabel("Time [sec]")
        ax4.set_ylabel("Velocity [m/s]")
        ax4.legend()

        # Button to start the process
        ax_button_use = plt.axes([0.68, 0.025, 0.1, 0.04])
        button_use = Button(ax_button_use, "Use", color="lightblue", hovercolor="0.975")
        ax_button_not_use = plt.axes([0.8, 0.025, 0.1, 0.04])
        button_not_use = Button(ax_button_not_use, "Not use", color="lightblue", hovercolor="0.975")

        def callback_button_use_clicked(event):
            filename = title + ".png"
            fig.savefig(os.path.join(self.output_dir_use, filename))

            self.update_acceleration_results(acceleration_result, mode, title, use_to_update=True)

            plt.close(fig)

        def callback_button_not_use_clicked(event):
            filename = title + ".png"
            fig.savefig(os.path.join(self.output_dir_not_use, filename))

            self.update_acceleration_results(acceleration_result, mode, title, use_to_update=False)

            plt.close(fig)

        button_use.on_clicked(callback_button_use_clicked)
        button_not_use.on_clicked(callback_button_not_use_clicked)

        plt.show()

    def report(self):
        pprint.pprint(self.acceleration_results)
        pprint.pprint(self.acceleration_results_each_file)

        buffer = ""

        buffer += "## Result\n\n"
        buffer += "### Acceleration\n\n"
        buffer += self.get_result_table("accel")
        buffer += "\n\n### Deceleration (Main brake)\n\n"
        buffer += self.get_result_table("brake")
        #buffer += "\n\n### Deceleration (Sub brake)\n\n"
        #buffer += self.get_result_table("sub_brake")

        filename = f"{self.vehicle_name}_map_accuracy_report.md"
        with open(os.path.join(self.report_path, filename), "w") as f:
            f.write(buffer)

    def get_result_table(self, mode):
        sample_key = list(self.acceleration_results[mode].keys())[0]
        headers = [""] + [f"{ref:.2f}" for ref, _ in self.acceleration_results[mode][sample_key]]
        table_data = []
        for target, values in self.acceleration_results[mode].items():
            row = [f"{target:.1f}"]
            for ref, actual in values:
                criteria_step = abs(target * CRITERIA_RANGE)
                if criteria_step < CRITERIA_MIN_ACCEL:
                    criteria_step = CRITERIA_MIN_ACCEL
                criteria_max = target + criteria_step
                criteria_min = target - criteria_step
                status = "OK" if criteria_min <= actual <= criteria_max else "NG"
                cell = (
                    f"{status}<br><br>"
                    f"criteria_max: {criteria_max:.2f}<br>"
                    f"criteria_min: {criteria_min:.2f}<br>"
                    f"actual: {actual:.2f}"
                )
                row.append(cell)
            table_data.append(row)

        markdown_table = tabulate(table_data, headers, tablefmt="github")
        return markdown_table

    def calculate_acceleration_from_velocity(self, topics, mode):
        stamp_autonomous_start, stamp_autonomous_end = lib.map.get_autonomous_mode_stamps(topics)

        target_acceleration = lib.map.get_target_acceleration(topics, mode)
        stamp_acceleration_start, stamp_acceleration_end = lib.map.get_acceleration_stamps(
            topics, mode, target_acceleration, stamp_autonomous_start, stamp_autonomous_end
        )

        stamps_for_calculating_acceleration = lib.map.get_stamps_for_calculating_acceleration(
            self.velocity_for_calculating_acceleration,
            topics,
            stamp_acceleration_start,
            stamp_acceleration_end,
            VELOCITY_MARGIN_KPH,
            NEAREST_VELOCITY_THRESHOLD_KPH,
        )

        acceleration_result = {target_acceleration: []}
        calculated_acceleration_time = []
        calculated_acceleration_mps2 = []
        map_axis_mps = np.array(MAP_AXIS_VELOCITY_KPH) * 1000.0 / 60.0 / 60.0
        for t0, t1, v0, v1, v_target in zip(
            stamps_for_calculating_acceleration[:-1],
            stamps_for_calculating_acceleration[1:],
            self.velocity_for_calculating_acceleration[:-1],
            self.velocity_for_calculating_acceleration[1:],
            map_axis_mps[1:],
        ):  
            if t0 == 0 or t1 == 0:
                acceleration_result[target_acceleration].append((round(v_target, 2), 0.0))
                continue

            diff_velocity = v1 - v0
            diff_stamp = t1 - t0
            acceleration = diff_velocity / diff_stamp

            acceleration_result[target_acceleration].append(
                (round(v_target, 2), round(acceleration, 2))
            )
            calculated_acceleration_time.append(round(np.average([t0, t1]), 3))
            calculated_acceleration_mps2.append(round(acceleration, 2))

            print("acceleration : ", acceleration)

        pprint.pprint(acceleration_result)
        pprint.pprint(calculated_acceleration_time)
        pprint.pprint(calculated_acceleration_mps2)

        plot_info = {}
        plot_info["stamp_autonomous_start"] = stamp_autonomous_start
        plot_info["stamp_autonomous_end"] = stamp_autonomous_end
        plot_info["target_acceleration"] = target_acceleration
        plot_info["stamp_acceleration_start"] = stamp_acceleration_start
        plot_info["stamp_acceleration_end"] = stamp_acceleration_end
        plot_info["stamps_for_calculating_acceleration"] = stamps_for_calculating_acceleration
        plot_info[
            "velocity_for_calculating_acceleration"
        ] = self.velocity_for_calculating_acceleration
        non_zero_indices = np.array(stamps_for_calculating_acceleration) != 0
        stamps_for_calculating_acceleration_non_zero = np.array(
            stamps_for_calculating_acceleration
        )[non_zero_indices]
        velocity_for_calculating_acceleration_non_zero = np.array(
            self.velocity_for_calculating_acceleration
        )[non_zero_indices]
        plot_info[
            "stamps_for_calculating_acceleration_non_zero"
        ] = stamps_for_calculating_acceleration_non_zero
        plot_info[
            "velocity_for_calculating_acceleration_non_zero"
        ] = velocity_for_calculating_acceleration_non_zero

        return (
            acceleration_result,
            calculated_acceleration_time,
            calculated_acceleration_mps2,
            plot_info,
        )

    def update_acceleration_results(self, acceleration_result, mode, title, use_to_update):
        target_acceleration = list(acceleration_result.keys())[0]
        self.acceleration_results_each_file[mode][title] = acceleration_result[target_acceleration]

        if use_to_update:
            if not self.acceleration_results[mode].get(target_acceleration):
                self.acceleration_results[mode][target_acceleration] = acceleration_result[
                    target_acceleration
                ]
            else:
                result = []
                for (v, a), (v_new, a_new) in zip(
                    self.acceleration_results[mode][target_acceleration],
                    acceleration_result[target_acceleration],
                ):
                    if v != v_new:
                        print(f"velocity axis is not same.")
                        sys.exit(1)
                    if a_new == 0:
                        result.append((v, a))
                        continue
                    if a == 0:
                        result.append((v, a_new))
                        continue
                    result.append((v, round(a + a_new / 2.0, 2)))
                self.acceleration_results[mode][target_acceleration] = result


def main():
    print("===== Map accuracy checker begin! =====")
    parser = argparse.ArgumentParser(description="Map accuracy checker")
    parser.add_argument("path", help="path to data")
    parser.add_argument("report_path", help="path to report")
    parser.add_argument("vehicle_name", help="vehicle name")
    args = parser.parse_args()
    print(f"Data path: {args.path}")
    print(f"Report path: {args.report_path}")

    checker = MapAccuracyChecker(args.path, args.report_path, args.vehicle_name)
    checker.run()


if __name__ == "__main__":
    main()
