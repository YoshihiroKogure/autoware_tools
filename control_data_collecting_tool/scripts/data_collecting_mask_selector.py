#!/usr/bin/env python3

from data_collecting_base_node import DataCollectingBaseNode
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from std_msgs.msg import Int32MultiArray
from publish_Int32MultiArray import publish_Int32MultiArray
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
import signal

class DataCollectingMaskSelector(DataCollectingBaseNode):
    def __init__(self):
        super().__init__("data_collecting_mask_selector")
        self.grid_update_time_interval = 5.0
        self.timer_plotter = self.create_timer(
            self.grid_update_time_interval,
            self.timer_callback_plotter,
        )
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([self.v_min, self.v_max])
        self.ax.set_ylim([self.a_max, self.a_min])
        self.ax.set_xticks(self.v_bins)
        self.ax.set_yticks(self.a_bins[::-1])
        self.ax.grid(True, color='black', linestyle='-', linewidth=1.5)
        self.ax.set_title("Data Collecting Mask Selector")
        self.ax.set_xlabel("Velocity (v)")
        self.ax.set_ylabel("Acceleration (a)")
        plt.ion()
        plt.show(block=False)

        try:
            self.fig.canvas.manager.window.attributes("-topmost", 0)
        except AttributeError:
            pass

        self.clicked_cells = {}
        self.dragging = False
        self.select_mode = True
        self.lock = threading.Lock()
        self.mask_vel_acc_publisher_ = self.create_publisher(
           Int32MultiArray, "/control_data_collecting_tools/mask_of_vel_acc", 10
        )
        self.mask_vel_acc = self.collected_data_counts_of_vel_acc 
        del self.collected_data_counts_of_vel_acc

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        self.dragging = True
        v_bin, a_bin, cell_center = self.get_cell_from_event(event)
        if cell_center:
            with self.lock:
                self.select_mode = cell_center not in self.clicked_cells
        self.update_cell(event)

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if self.dragging:
            self.update_cell(event)

    def get_cell_from_event(self, event):
        if event.xdata is None or event.ydata is None:
            return None, None, None

        v_bin = np.digitize(event.xdata, self.v_bins) - 1
        a_bin = np.digitize(event.ydata, self.a_bins) - 1
        
        if 0 <= v_bin < self.num_bins_v and 0 <= a_bin < self.num_bins_a:
            cell_x = self.v_bin_centers[v_bin]
            cell_y = self.a_bin_centers[a_bin]
            cell_center = (cell_x, cell_y)
            return v_bin, a_bin, cell_center
        else:
            return None, None, None

    def update_cell(self, event):
        v_bin, a_bin, cell_center = self.get_cell_from_event(event)
        if cell_center:
            with self.lock:
                if self.select_mode:
                    if cell_center not in self.clicked_cells:
                        rect = patches.Rectangle(
                            (cell_center[0]- (self.v_bins[1]-self.v_bins[0]) / 2, cell_center[1]- (self.a_bins[1]-self.a_bins[0]) / 2),
                            self.v_bins[1] - self.v_bins[0],
                            self.a_bins[1] - self.a_bins[0],
                            linewidth=1, edgecolor='r', facecolor='r'
                        )
                        self.ax.add_patch(rect)
                        self.clicked_cells[cell_center] = rect
                else:
                    if cell_center in self.clicked_cells:
                        rect = self.clicked_cells[cell_center]
                        rect.remove()
                        del self.clicked_cells[cell_center]

        self.fig.canvas.draw_idle()

    def timer_callback_plotter(self):
        with self.lock:
            self.mask_vel_acc.fill(0)
            for cell in self.clicked_cells:
                v_bin = np.digitize(cell[0], self.v_bins) - 1
                a_bin = np.digitize(cell[1], self.a_bins) - 1
                if 0 <= v_bin < self.num_bins_v and 0 <= a_bin < self.num_bins_a:
                    self.mask_vel_acc[v_bin, a_bin] = 1

            publish_Int32MultiArray(
                self.mask_vel_acc_publisher_, self.mask_vel_acc
            )

def main(args=None):
    rclpy.init(args=args)
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)

    data_collecting_mask_selector = DataCollectingMaskSelector()
    executor = MultiThreadedExecutor()
    executor.add_node(data_collecting_mask_selector)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        while rclpy.ok() and not shutdown_event.is_set():
            plt.pause(0.05)
    finally:
        data_collecting_mask_selector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
