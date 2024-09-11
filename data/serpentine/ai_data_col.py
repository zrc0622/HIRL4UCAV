from environments import dogfight_client as df
import time
from data.serpentine.ai_env import *
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

def plot_trajectories(ally_pos, enemy_pos, save_path=None):
    # 提取x和z坐标
    ally_x, ally_y, ally_z = zip(*ally_pos) if ally_pos else ([], [])
    enemy_x, enemy_y, enemy_z = zip(*enemy_pos) if enemy_pos else ([], [])

    # 创建图形和轴
    plt.figure(figsize=(10, 10))

    # 绘制友军和敌军轨迹
    plt.plot(ally_x, ally_z, 'b-o', label='Ally Trajectory', markersize=1)  # 蓝色线表示友军
    plt.plot(enemy_x, enemy_z, 'r-o', label='Enemy Trajectory', markersize=1)  # 红色线表示敌军

    # 设置图例、标题和坐标轴标签
    plt.legend()
    plt.title('Aircraft Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # 显示网格
    plt.grid(True)

    # 如果提供了保存路径，则保存图像
    if save_path:
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        plt.savefig(save_path)
        print(f"图像已保存至 {save_path}")

df.connect("172.30.58.126", 11111) # 连接环境
time.sleep(2) # 等待连接
df.disable_log() # 关闭日志
render = True
df.set_renderless_mode(render) # 开启渲染模式
df.set_client_update_mode(True) # 确保数据传输正确
planes = df.get_planes_list()

plane_id = planes[0]
oppo_id = planes[3]

env = AIHarfangSerpentineEnv()
env.Plane_ID_ally = plane_id # ally 1
env.Plane_ID_oppo = oppo_id # ennemy_2

state = env.reset()
df.update_scene()
state_dim = state.shape[0]
action_dim = 4
state_list = []
action_list = []
delt_list = []

episode = 1
step = 0

invalid_data = 0
while episode <= 50:
    env.reset()
    df.activate_IA(plane_id)
    health = env._get_health()
    
    episode_step = 0
    lock = 0
    fire = 0
    temp_state_list = [] # 通过临时state-action删去无效轨迹
    temp_action_list = []  

    ally_pos = []
    ennemy_pos = [] 

    while health > 0:
        # time.sleep(1/80)
        df.update_scene()
        
        state = env._get_observation()
        temp_state_list.append(state) 
        action = env._get_action()
        temp_action_list.append(action)

        env.set_ennemy_yaw()

        pos = env.get_pos()
        ally_pos.append(pos)
        pos = env.get_oppo_pos()
        ennemy_pos.append(pos)
       
        if state[-1]>0 and state[-2]>0:
            if lock ==0:
                lock = episode_step
                print('can step:{}'.format(episode_step))

        if action[-1]>0:
            print('fire step:{}'.format(episode_step))
            if fire ==0:
                fire = episode_step

        health = env._get_health()
        step += 1
        episode_step += 1
    
    if episode_step > 2500 or fire-lock>500 or fire-lock<0:
        invalid_data += 1
        step -= episode_step

    else:
        plot_trajectories(ally_pos, ennemy_pos, save_path=f"data/serpentine/plot/{episode}.")
        state_list.extend(temp_state_list)
        action_list.extend(temp_action_list)
        delt_list.append(fire-lock)
        print("read")

    print(f"episode = {episode}  step = {episode_step}  delt = {fire-lock}")
    episode += 1


print("episode", episode, " step", step, " state dim", state_dim, " invalid data", invalid_data)
print(delt_list)
action_array = np.array(action_list)
state_array = np.array(state_list)
print(action_array.shape)
data = [state_array, action_array]

filename = 'data/serpentine/expert_data_ai.csv'

# with open(filename, 'wb') as file: # 打开pickle文件
#     pickle.dump(data, file) # 写入pickle文件

with open(filename, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
    writer = csv.writer(file)
    writer.writerows(data)  # 将数据写入CSV文件
print("ok")
df.set_client_update_mode(False)
df.disconnect()
