import dogfight_client as df
import time
from DemoEnv import *
import pickle
import numpy as np

df.connect("10.241.58.126", 50888) # 连接环境
time.sleep(2) # 等待连接
df.disable_log() # 关闭日志？
render = False
df.set_renderless_mode(render) # 开启渲染模式
df.set_client_update_mode(True) # 通过update_scene()更新场景
planes = df.get_planes_list()

plane_id = planes[0]
oppo_id = planes[3]

env = DemoEnv()
env.Plane_ID_ally = plane_id
env.Plane_ID_oppo = oppo_id

state = env.reset()
df.update_scene()
state_dim = state.shape[0]
action_dim = 4
state_list = []
action_list = []

df.activate_IA(plane_id)
t = 0
while t < 6000:
    time.sleep(1/20)
    df.update_scene()
time.sleep(60)

# 直线飞
t = 0
while t < 60:
    time.sleep(1/20)
    state = env._get_observation()
    df.update_scene()
    action = np.array([0, 0, 0, -1])
    action_list.append(action)
    state_list.append(state) 
    t = t + 1   
df.retract_gear(plane_id)

# 抬升
df.set_plane_pitch(plane_id, -0.5)
p = 0
while p < 40:
	time.sleep(1/20)
	state = env._get_observation()
	plane_state = df.get_plane_state(plane_id)
	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
	df.update_scene()
	p = plane_state["pitch_attitude"]
	action = np.array([-0.5, 0, 0, -1])
	action_list.append(action)
	state_list.append(state)

# 保持
df.set_plane_pitch(plane_id, 0)
t = 0
while t < 640:
    time.sleep(1/20)
    state = env._get_observation()
    df.update_scene()
    action = np.array([0, 0, 0, -1])
    action_list.append(action)
    state_list.append(state) 
    t = t + 1

# 低头
df.set_plane_pitch(plane_id, 0.5)
while p > 2:
	time.sleep(1/20)
	state = env._get_observation()
	plane_state = df.get_plane_state(plane_id)
	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
	df.update_scene()
	p = plane_state["pitch_attitude"]
	action = np.array([0.5, 0, 0, -1])
	action_list.append(action)
	state_list.append(state)

# 保持
df.set_plane_pitch(plane_id, 0)
locked = False	
while locked == False:
	time.sleep(1/20)
	state = env._get_observation()
	plane_state = df.get_plane_state(plane_id)
	df.update_scene()
	locked = plane_state["target_locked"]
	action = np.array([0, 0, 0, -1])
	action_list.append(action)
	state_list.append(state)
	
# time.sleep(1/20)
# state = env._get_observation() # 状态
# df.fire_missile(plane_id, 0) # 动作
# df.update_scene() # 运用动作
# action = np.array([0, 0, 0, 1])
# action_list.append(action)
# state_list.append(state)

m = 0
while m < 100:
    time.sleep(1/20)
    missile_slot = m//30
    state = env._get_observation() # 状态
    df.fire_missile(plane_id, missile_slot) # 动作
    print(missile_slot)
    df.update_scene() # 运用动作
    action = np.array([0, 0, 0, missile_slot+1])
    action_list.append(action)
    state_list.append(state)
    m += 1

t = 0
while t < 300:
    time.sleep(1/20)
    state = env._get_observation()
    df.update_scene()
    action = np.array([0, 0, 0, -1])
    action_list.append(action)
    state_list.append(state) 
    t = t + 1   

action_array = np.array(action_list)
state_array = np.array(state_list)
data = [state_array, action_array]

filename = 'expert_data.pkl'

with open(filename, 'wb') as file: # 打开pickle文件
    pickle.dump(data, file) # 写入pickle文件

df.set_client_update_mode(False)
df.disconnect()