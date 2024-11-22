import yaml
import csv
import time
import random
import numpy as np
from hirl.environments import dogfight_client as df
from hirl.data.straight_line.ai_env import *
from hirl.utils.plot import plot_2d_trajectories

with open('local_config.yaml', 'r') as file:
    local_config = yaml.safe_load(file)

df.connect(local_config["network"]["ip"], 13579) # 连接环境
time.sleep(2) # 等待连接
df.disable_log() # 关闭日志
df.set_renderless_mode(True) # 关闭渲染模式
df.set_client_update_mode(True)

planes = df.get_planes_list()
plane_id = planes[0]
oppo_id = planes[3]

env = AIHarfangEnv()
env.Plane_ID_ally = plane_id # ally 1
env.Plane_ID_oppo = oppo_id # ennemy_2


state_dim = 19
action_dim = 4
state_list = []
action_list = []
delt_list = []

episode = 1
step = 0
invalid_data = 0

while episode <= 10:
    env.reset()
    df.activate_IA(plane_id)
    health = 1
    
    episode_step = 0
    lock = 0
    fire = 0
    temp_state_list = []
    temp_action_list = []
    done = False

    ally_pos = []
    ennemy_pos = [] 

    while health > 0:
        health = env._get_health()
        
        state = env._get_observation()
        temp_state_list.append(state) 
        
        if state[10]>0 and state[11]>0:
            probability = random.random()
            if probability < 0.3 and not done:
                env.fire()
                done = True

            if lock == 0:
                lock = episode_step
                print('can step:{}'.format(episode_step))
        df.update_scene()

        action = env._get_action()
        temp_action_list.append(action)

        pos = env.get_pos()
        ally_pos.append(pos)
        pos = env.get_oppo_pos()
        ennemy_pos.append(pos)

        if action[-1]>0:
            print('fire step:{}'.format(episode_step))
            if fire == 0:
                fire = episode_step

        step += 1
        episode_step += 1
    
    if episode_step > 2000 or fire-lock>20 or fire-lock<0:
        invalid_data += 1
        step -= episode_step

    else:
        state_list.extend(temp_state_list)
        action_list.extend(temp_action_list)
        delt_list.append(fire-lock)
        print("read")
    plot_2d_trajectories(ally_pos, ennemy_pos, save_path=f"hirl/data/straight_line/plot/{episode}.png")

    print(f"episode = {episode}  step = {episode_step}  delt = {fire-lock}")
    episode += 1


print("episode", episode, " step", step, " invalid data", invalid_data)
print(delt_list)
action_array = np.array(action_list)
state_array = np.array(state_list)
data = [state_array, action_array]

filename = 'hirl/data/straight_line/expert_data_ai_small.csv'

with open(filename, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
    writer = csv.writer(file)
    writer.writerows(data)  # 将数据写入CSV文件

print("Finish")
df.set_client_update_mode(False)
df.disconnect()
