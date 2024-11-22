import pandas as pd
import numpy as np
import csv

def read_data(data_dir):
    expert_data = pd.read_csv(data_dir, header=None)
    state = expert_data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ') for item in state])  # 将state变为(?, 14)的格式，一行代表一个state
    action = expert_data.iloc[1].to_numpy()
    npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
    return npstate, npaction

def write_data(npstate, npaction, data_dir):
    data = [npstate, npaction]
    with open(data_dir, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
        writer = csv.writer(file)
        writer.writerows(data)  # 将数据写入CSV文件

def up_sample(BCActions):
    target_indices = np.where(BCActions[:, 3] == 1)[0]
    return target_indices
