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

#ifndef DATA_COLLECTING_GOAL_POSE_HPP_
#define DATA_COLLECTING_GOAL_POSE_HPP_

#include <rviz_common/tool.hpp>
#include <rviz_default_plugins/tools/goal_pose/goal_tool.hpp>


namespace rviz_plugins
{
class DataCollectingGoalPose : public rviz_default_plugins::tools::GoalTool
{
  Q_OBJECT
public:
  DataCollectingGoalPose();
  ~DataCollectingGoalPose();

  void onInitialize() override;

  double x;
};
}  // namespace rviz_plugins

#endif  // DATA_COLLECTING_GOAL_POSE_HPP_
