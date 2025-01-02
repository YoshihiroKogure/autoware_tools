import numpy as np
import cvxpy as cp
import pandas as pd
import argparse
import os 

VELOCITY_AXIS = [ 0,1.39,	2.78,	4.17,	5.56,	6.94,	8.33,	9.72,	11.11,	12.5, 13.89]
ACCEL_AXIS = [ i * 0.1 for i in range(6) ]
BRAKE_AXIS = [ i * 0.1 for i in range(9) ]
Not_Modified_Cell_Weight = 1e8
diff_of_cell = 0.01

def optimize_map(matrix, modified_idx):
    m, n = matrix.shape
    # 最適化変数
    B = cp.Variable((m, n))
    # 目的関数: 元データとの差を最小化
    objective = cp.sum_squares(B - matrix)#cp.Minimize(cp.sum_squares(B - matrix))
    # 制約: 行・列の単調性を強制
    constraints = []
    for i in range(m):
        for j in range(n - 1):
            constraints.append(B[i, j+ 1] + diff_of_cell <= B[i, j])  # 行方向
    for i in range(m - 1):
        for j in range(n):
            constraints.append(B[i+1, j] + diff_of_cell <= B[i, j])  # 列方向
    
    for i in range(m):
        for j in range(n - 1):
            if (i,j) not in modified_idx:
                objective += Not_Modified_Cell_Weight * cp.sum_squares(B[i,j] - matrix[i,j])
    objective = cp.Minimize(objective)
    # 問題を定義
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return B.value

def compute_row_neighbors_average(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix, dtype=float)  # 結果用の行列を作成
    modified_idx = []
    for i in range(rows):
        for j in range(cols):
            # 近傍要素のリスト
            if abs(matrix[i,j]) < 1e-6:
                neighbors = []
                # 上
                #if i > 0 and abs(matrix[i-1, j]) > 1e-6:
                    #neighbors.append(matrix[i-1, j])
                # 下
                #if i < rows-1 and abs(matrix[i+1, j]) > 1e-6:
                    #neighbors.append(matrix[i+1, j])
                # 左
                if j > 0 and abs(matrix[i, j-1]) > 1e-6:
                    neighbors.append(matrix[i, j-1])
                # 右
                if j < cols-1 and abs(matrix[i, j+1]) > 1e-6:
                    neighbors.append(matrix[i, j+1])
                
                # 平均を計算（近傍が存在する場合のみ）
                if neighbors:
                    matrix[i, j] = sum(neighbors) / len(neighbors)
                    modified_idx.append((i,j))
                else:
                    matrix[i, j] = 0.0  # 近傍がない場合は 0.0 にする（任意の値に調整可能

    return matrix, modified_idx

def compute_col_neighbors_average(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix, dtype=float)  # 結果用の行列を作成
    modified_idx = []
    for i in range(rows):
        for j in range(cols):
            # 近傍要素のリスト
            if abs(matrix[i,j]) < 1e-6:
                neighbors = []
                # 上
                if i > 0 and abs(matrix[i-1, j]) > 1e-6:
                    neighbors.append(matrix[i-1, j])
                # 下
                if i < rows-1 and abs(matrix[i+1, j]) > 1e-6:
                    neighbors.append(matrix[i+1, j])
                
                # 平均を計算（近傍が存在する場合のみ）
                if neighbors:
                    matrix[i, j] = sum(neighbors) / len(neighbors)
                    modified_idx.append((i,j))
                else:
                    matrix[i, j] = 0.0  # 近傍がない場合は 0.0 にする（任意の値に調整可能
    return matrix, modified_idx

def main():
    parser = argparse.ArgumentParser(description="Accel and brake map generator")
    parser.add_argument("map_path", help="path to map")
    args = parser.parse_args()
    accel_map_ = pd.read_csv(args.map_path + "/accel_map.csv", delimiter=",", header=0)
    accel_map = np.zeros((len(ACCEL_AXIS), len(VELOCITY_AXIS)))
    _accel_m, _accel_n = accel_map_.to_numpy()[:,1:].shape
    accel_map[:_accel_m,:_accel_n] = accel_map_.to_numpy()[:,1:]


    brake_map_ = pd.read_csv(args.map_path + "/brake_map.csv", delimiter=",", header=0)
    brake_map = np.zeros((len(BRAKE_AXIS), len(VELOCITY_AXIS)))
    _brake_m, _brake_n = brake_map_.to_numpy()[:,1:].shape
    brake_map[:_brake_m,:_brake_n] = brake_map_.to_numpy()[:,1:]


    accel_m, accel_n = accel_map.shape
    brake_m, brake_n = brake_map.shape

    map_ = np.zeros((accel_m + brake_m - 1, accel_n))

    map_[:accel_m, :] = accel_map[::-1,:]
    for i in range(accel_n):
        component = []
        if abs(accel_map[0,i]) > 1e-6:
            component.append(accel_map[0,i])
        if abs(brake_map[0,i]) > 1e-6:
            component.append(brake_map[0,i])
        if component:
            map_[accel_m-1, i] = np.mean(component)
    map_[accel_m:,:] = brake_map[1:,:]

    modified_idx = []
    for i in range(len(map_[0,:])):
        map_, modified_idx_row = compute_row_neighbors_average(map_)
        modified_idx += modified_idx_row
    
    for j in range(len(brake_map)):
        map_, modified_idx_col = compute_col_neighbors_average(map_)
        modified_idx += modified_idx_col

    map_ = optimize_map(map_, modified_idx)
    

    modified_accel_map_ = map_[:accel_m, :][::-1,:]
    modified_accel_map = np.zeros((len(modified_accel_map_)+1, len(modified_accel_map_[0])+1))
    modified_accel_map[1:,1:] = modified_accel_map_
    modified_accel_map[0,1:] = np.array([np.round(i*1.39,2) for i in range(len(modified_accel_map_[0]))])
    modified_accel_map[1:,0] = np.array(ACCEL_AXIS)


    modified_brake_map_ = map_[accel_m-1:, :]
    modified_brake_map = np.zeros((len(modified_brake_map_)+1, len(modified_brake_map_[0])+1))
    modified_brake_map[1:,1:] = modified_brake_map_
    modified_brake_map[0,1:] = np.array([np.round(i*1.39,2) for i in range(len(modified_brake_map_[0]))])
    modified_brake_map[1:,0] = np.array(BRAKE_AXIS)

    directory_path = "modified_map"
    os.makedirs(directory_path, exist_ok=True)

    modified_accel_map = np.round(modified_accel_map,decimals=3).astype('<U10')
    modified_brake_map = np.round(modified_brake_map,decimals=3).astype('<U10')

    modified_accel_map[0,0] = "modified"
    modified_brake_map[0,0] = "modified"
    
    np.savetxt("modified_map" + "/accel_map.csv", modified_accel_map, delimiter=",",fmt="%s")
    np.savetxt("modified_map" + "/brake_map.csv", modified_brake_map, delimiter=",",fmt="%s")


if __name__ == "__main__":
    main()