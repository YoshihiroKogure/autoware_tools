// Copyright 2024 Proxima Technology Inc, TIER IV Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "data_collecting_goal_pose.hpp"

namespace rviz_plugins
{
DataCollectingGoalPose::DataCollectingGoalPose()
{
  // skip
  x = 0.0;
}
DataCollectingGoalPose::~DataCollectingGoalPose() = default;
void DataCollectingGoalPose::onInitialize()
{
  GoalTool::onInitialize();
  setName("DataCollectingGoalPose");
}
} // namespace rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_plugins::DataCollectingGoalPose, rviz_common::Tool)
