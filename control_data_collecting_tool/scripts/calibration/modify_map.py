import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def optimize_accel(matrix):
    m, n = matrix.shape
    # 最適化変数
    B = cp.Variable((m, n))
    # 目的関数: 元データとの差を最小化
    objective = cp.Minimize(cp.sum_squares(B - matrix))
    # 制約: 行・列の単調性を強制
    constraints = []
    for i in range(m):
        for j in range(n - 1):
            constraints.append(B[i, j + 1] <= B[i, j])  # 行方向
    for i in range(m - 1):
        for j in range(n):
            constraints.append(B[i, j] <= B[i + 1, j])  # 列方向
    # 問題を定義
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return B.value

def optimize_brake(matrix):
    m, n = matrix.shape
    # 最適化変数
    B = cp.Variable((m, n))
    # 目的関数: 元データとの差を最小化
    objective = cp.Minimize(cp.sum_squares(B - matrix))
    # 制約: 行・列の単調性を強制
    constraints = []
    for i in range(m):
        for j in range(n - 1):
            constraints.append(B[i, j+ 1] <= B[i, j])  # 行方向
    for i in range(m - 1):
        for j in range(n):
            constraints.append(B[i+1, j] <= B[i, j])  # 列方向
    # 問題を定義
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return B.value

def compute_row_neighbors_average(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix, dtype=float)  # 結果用の行列を作成

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
                else:
                    matrix[i, j] = 0.0  # 近傍がない場合は 0.0 にする（任意の値に調整可能

    return matrix

def compute_col_neighbors_average(matrix):
    rows, cols = matrix.shape
    result = np.zeros_like(matrix, dtype=float)  # 結果用の行列を作成

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
                else:
                    matrix[i, j] = 0.0  # 近傍がない場合は 0.0 にする（任意の値に調整可能）7
    return matrix

def main():
    parser = argparse.ArgumentParser(description="Accel and brake map generator")
    parser.add_argument("map_ath", help="path to map")
    args = parser.parse_args()
    accel_map_ = pd.read_csv(args.map_path + "/accel_map.csv", delimiter=",", header=0)
    accel_map = accel_map_.to_numpy()[:,1:]
    accel_map

    brake_map_ = pd.read_csv(args.map_path + "/brake_map.csv", delimiter=",", header=0)
    brake_map = brake_map_.to_numpy()[:,1:]


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

    for i in range(len(map_[0,:])):
        map_ = compute_row_neighbors_average(map_)
    for j in range(len(brake_map)):
        map_ = compute_col_neighbors_average(map_)
    map_ = optimize_brake(map_)

    modified_accel_map = map_[:accel_m, :]
    modified_brake_map = map_[accel_m-1:, :]

    np.savetxt(args.map_path + "/modified_accel_map.csv", modified_accel_map[::-1,:],delimiter=",")
    np.savetxt(args.map_path + "/modified_brake_map.csv", modified_brake_map,delimiter=",")


if __name__ == "__main__":
    main()