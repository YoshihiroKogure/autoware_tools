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

from std_msgs.msg import Int32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout

def publish_Int32MultiArray(publisher_, array_data):
    msg = Int32MultiArray()

    flattened_data = array_data.flatten().tolist()
    msg.data = flattened_data

    layout = MultiArrayLayout()

    dim1 = MultiArrayDimension()
    dim1.label = "rows"
    dim1.size = len(array_data)
    dim1.stride = len(flattened_data)

    dim2 = MultiArrayDimension()
    dim2.label = "cols"
    dim2.size = len(array_data[0])
    dim2.stride = len(array_data[0])

    layout.dim.append(dim1)
    layout.dim.append(dim2)

    msg.layout = layout

    publisher_.publish(msg)