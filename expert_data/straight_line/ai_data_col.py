import environment.dogfight_client as df
import time
from expert_data.straight_line.ai_env import *
import numpy as np
import csv

df.connect("10.241.58.126", 22222) # 连接环境
time.sleep(2) # 等待连接
df.disable_log() # 关闭日志
render = True
df.set_renderless_mode(render) # 开启渲染模式
df.set_client_update_mode(True) # 确保数据传输正确
planes = df.get_planes_list()

plane_id = planes[0]
oppo_id = planes[3]

env = AIHarfangEnv()
env.Plane_ID_ally = plane_id
env.Plane_ID_oppo = oppo_id

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
while episode <= 100:
    env.reset()
    df.activate_IA(plane_id)
    health = env._get_health()
    
    episode_step = 0
    lock = 0
    fire = 0
    temp_state_list = [] # 通过临时state-action删去无效轨迹
    temp_action_list = []   

    while health > 0:
        # time.sleep(1/80)
        df.update_scene()
        
        state = env._get_observation()
        temp_state_list.append(state) 
        action = env._get_action()
        temp_action_list.append(action)
       
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
        
    if episode_step > 1500 or fire-lock>35 or fire-lock<0:
        invalid_data += 1
        step -= episode_step

    else:
        state_list.extend(temp_state_list)
        action_list.extend(temp_action_list)
        delt_list.append(fire-lock)
        print("read")

    print(f"episode = {episode}  step = {episode_step}  delt = {fire-lock}")

    episode += 1

print("episode", episode, " step", step, " state dim", state_dim, " invalid date", invalid_data)
print(delt_list)
action_array = np.array(action_list)
state_array = np.array(state_list)
print(action_array.shape)
data = [state_array, action_array]

filename = 'expert_data/straight_line/expert_data_ai2.csv'

with open(filename, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
    writer = csv.writer(file)
    writer.writerows(data)  # 将数据写入CSV文件
print("ok")
df.set_client_update_mode(False)
df.disconnect()