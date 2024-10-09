#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import serialize_message
import os
from datetime import datetime
import time
from functools import partial

class DataCollectingRosbagRecord(Node):
    def __init__(self, topics):
        super().__init__('data_collecting_rosbag_record')
        self.topics = topics
        self.writer = SequentialWriter()
        self.subscribed = False
        self.start_record = False

        self.timer_period_callback = 1.0
        self.timer_callback = self.create_timer(
            self.timer_period_callback,
            self.record_message,
        )

    def subscription(self):
        topic_type_list = []
        unsubscribed_topic = []

        for topic_name in self.topics:
            topic_type = self.get_topic_type(topic_name)
            topic_type_list.append(topic_type)
            if topic_type is None:
                unsubscribed_topic.append(topic_name)

        if len(unsubscribed_topic) > 0:
            self.get_logger().info(f"Failed to get type for topic: {unsubscribed_topic}")
            return

        now = datetime.now()
        formatted_time = now.strftime("%Y_%m_%d-%H_%M_%S")
        bag_dir = f'./rosbag2_{formatted_time}'
        storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        try:
            self.writer.open(storage_options, converter_options)
        except Exception as e:
            self.get_logger().error(f"Failed to open bag: {e}")
            return
        
        for topic_name, topic_type in zip(self.topics, topic_type_list):
            self.get_logger().info(f"Recording topic: {topic_name} of type: {topic_type}")
            topic_metadata = TopicMetadata(name=topic_name, type=topic_type, serialization_format='cdr')
            self.writer.create_topic(topic_metadata)
        self.subscribed = True

    def get_topic_type(self, topic_name):
        topic_names_and_types = self.get_topic_names_and_types()
        for name, types in topic_names_and_types:
            if name == topic_name:
                return types[0]
        return None

    def record_message(self):
        if not self.subscribed:
            self.subscription()
        else:
            if not self.start_record:
                for topic_name in self.topics:
                    topic_type = self.get_topic_type(topic_name)
                    if topic_type:
                        msg_module = get_message(topic_type)
                        self.create_subscription(
                            msg_module, topic_name, partial(self.write_message, topic_name), 10
                        )
                self.start_record = True

    def write_message(self, topic_name, message):
        try:
            serialized_message = serialize_message(message)
            current_time = rclpy.clock.Clock().now()
            self.writer.write(topic_name, serialized_message, current_time.nanoseconds)
        except Exception as e:
            self.get_logger().error(f"Failed to write message: {e}")

def main(args=None):
    rclpy.init(args=args)
    topics = ["/localization/kinematic_state", "/localization/acceleration"]

    data_collecting_rosbag_record = DataCollectingRosbagRecord(topics)
    rclpy.spin(data_collecting_rosbag_record)
    data_collecting_rosbag_record.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
