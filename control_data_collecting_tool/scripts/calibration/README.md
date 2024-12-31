## How to calibrate Accel/Brake map

-  Collect constant actuation command data for map generation
    Following the instruction [`external_actuation_cmd`: Collect constant actuation command data](../../README.md#external_actuation_cmd-collect-constant-actuation-command-data), please collect constant accel pedal input (0.0 ~ 0.5) data in `calibration_data/accel` and constant brake pedal input data ( 0.0 ~ 0.8 ) in `calibration_data/brake`. ROS bag files are saved where you run 

    ```bash
    ros2 run control_data_collecting_tool data_collecting_acceleration_cmd
    ```

-  Generate and modify map
    - Map generation
    ```bash
    python3 accel_brake_map_generator.py calibration_data map "default"
    ```
    `calibration_data` is directory where constant pedal input data are located and map is a directory where accel/brake maps are located.

    - Map modification: Modify map so that generated map satisfies certain conditions.
    ```bash
    python3 modify_map.py map
    ```
    By the command above, modified maps are generated in `modified_mp/`.

- Collect constant acceleration input data for map accuracy check

    Following the instruction [`external_acceleration_cmd`: Collect constant acceleration command data](../../README.md#external_acceleration_cmd-collect-constant-acceleration-command-data), please collect constant positive acceleration input data in `check_data/accel` and constant negative acceleration input data in `check_data/brake`. ROS bag files are saved where you run 

    ```bash
    ros2 run control_data_collecting_tool data_collecting_acceleration_cmd
    ```

- Run accuracy check scripts``

    ```bash
    python3 map_accuracy_checker.py check_data check_data "calibrated"
    ```


## Details of `modify_map.py`

1. Input Parsing and Initialization:

    The script reads acceleration (accel_map.csv) and braking (brake_map.csv) maps from the specified directory.
    These maps are represented as 2D arrays, initialized with zeros where dimensions exceed the input size.

2. Map Merging:

    Combines the acceleration and braking maps into a unified map. The top rows are the reversed acceleration map, while the bottom rows are the braking map. The merging point is computed by averaging overlapping values.

3. Neighbor Interpolation:

    For missing or zero values in the map, neighbor-based averaging is performed:
        Row-wise interpolation: Fills missing values using adjacent row elements.
        Column-wise interpolation: Fills missing values using adjacent column elements.
    Indices of modified cells are tracked for optimization weighting.

4. Optimization:

    The merged map is optimized using cvxpy to enforce monotonicity along rows and columns.
    Monotonicity constraints ensure that values increase or decrease by a minimum threshold (diff_of_cell).
    Cells that were not modified during interpolation are penalized with a large weight to preserve their original values.

5. Output Map Generation:

    The optimized map is split back into separate acceleration and braking maps.
    Row and column headers are added, representing velocity and acceleration/brake levels.
    The maps are rounded to three decimal places and saved as CSV files in the modified_map directory.
