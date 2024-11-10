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

from courses.figure_eight import Figure_Eight
from courses.straight_line_positive import Straight_Line_Positive
from courses.straight_line_negative import Straight_Line_Negative
from courses.u_shaped import U_Shaped
from courses.along_road import Along_Road

def load_course(course_name, step_size, params_dict):

    if course_name == "eight_course":
        course = Figure_Eight(step_size, params_dict)
    elif course_name == "straight_line_positive":
        course = Straight_Line_Positive(step_size, params_dict)
    elif course_name == "straight_line_negative":
        course = Straight_Line_Negative(step_size, params_dict)
    elif course_name == "u_shaped_return":
        course = U_Shaped(step_size, params_dict)
    elif course_name == "along_road":
        course = Along_Road(step_size, params_dict)

    return course
