import numpy as np
import hirl.environments.dogfight_client as df
from hirl.environments.constants import *
import gym
import os
import inspect
import random
import math
    
class HarfangEnv():
    def __init__(self):
        self.done = False
        self.loc_diff = 0
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), 
                                           dtype=np.float64)
        self.Plane_ID_oppo = "ennemy_2"  # Opponent aircrafts name
        self.Plane_ID_ally = "ally_1"  # our aircrafts name
        self.Aircraft_Loc = None
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.reward = 0
        self.Plane_Irtifa = 0
        self.now_missile_state = False # 导弹此时是否发射（本次step是否发射了导弹）
        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally) # 导弹列表
        self.missile1_id = self.missile[0] # 导弹1
        self.oppo_health = 0.2 # 敌机血量
        self.target_angle = None
        self.success = 0 # stepsuccess
        self.episode_success = False
        self.fire_success = False

    def reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.success = 0
        self.done = False
        self._reset_machine()
        self._reset_missile() # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally # 当前时刻状态

        return state_ally
    
    def random_reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False # 运用动作前是否锁敌
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.missile1_state = True # 运用动作前导弹1是否存在
        self.n_missile1_state = True # 运用动作后导弹1是否存在
        self.success = 0
        self.done = False
        self._random_reset_machine()
        self._reset_missile() # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally

        return state_ally

    def _random_reset_machine(self):
        df.reset_machine("ally_1") # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2 #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def step(self, action):
        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.success
    
    def step_test(self, action):
        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success
    
    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        self._get_loc_diff()  # get location difference information for reward
        
        # 距离惩罚：帮助追击
        self.reward -= (0.0001 * (self.loc_diff)) # 0.4

        # 目标角惩罚：帮助锁敌
        self.reward -= (self.target_angle)*10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4

        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        # 开火奖励：帮助开火
        if self.now_missile_state == True: # 如果此step导弹发射
            self.reward -= 8
            if self.missile1_state == True and self.Ally_target_locked == False: # 且导弹存在、不锁敌
                # self.reward -= 100 # 4、4
                self.success = -1
                print('failed to fire')
            elif self.missile1_state == True and self.Ally_target_locked == True: # 且导弹存在且锁敌
                # self.reward += 8 # 100、4
                print('successful to fire')
                self.success = 1
                self.fire_success = True
            else:
                # self.reward -= 10
                self.reward -= 0

        # 坠落奖励（最终奖励）
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600 # 无、无
            print('enemy have fallen')

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))
        
        # self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0) #
            self.now_missile_state = True # 此时导弹发射
            # print("fire")
        else:
            self.now_missile_state = False
        
        df.update_scene()
        
    def _get_termination(self):
        # if self.loc_diff < 200:
        #     self.done = True
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0: # 敌机血量低于0则结束
            self.done = True
            self.episode_success = True
        # if self.now_missile_state == True:
        #     self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1") # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2 #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self): #
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally) # 重新装填导弹

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) + ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) + ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** (1 / 2)

    def _get_observation(self): # 注意get的是n_state
        # Plane States
        Plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                     Plane_state["position"][1] / NormStates["Plane_position"],
                     Plane_state["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [Plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"], # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
                       Plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"], # 航向角，0 -> pai -> -pai -> 0
                       Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]] # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Plane_Heading = Plane_state["heading"] / NormStates["Plane_heading"] # 航向角，0 -> 360
        Plane_Pitch_Att = Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"] # 俯仰：俯0 -> -90，仰0 -> 90 
        Plane_Roll_Att = Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"] # 横滚角，顺时针：0 -> -90 ->0 -> 90 -> 0

        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]
        

        # Opponent States
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"], # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
                       Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"], # 航向角，0 -> pai -> -pai -> 0
                       Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]] # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        self.Plane_Irtifa = Plane_state["position"][1]
        self.Aircraft_Loc = Plane_state["position"]
        self.Oppo_Loc = Oppo_state["position"]

        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = Plane_state["target_locked"]
        if self.n_Ally_target_locked == True: # 
            locked = 1
        else:
            locked = -1

        target_angle = Plane_state['target_angle'] / 180
        self.target_angle = target_angle
        
        Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], Plane_Pos[1] - Oppo_Pos[1], Plane_Pos[2] - Oppo_Pos[2]]
        
        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        
        oppo_hea = self.oppo_health['health_level'] # 敌机初始血量为20

        # if self.now_missile_state == True:
        #     if_fire = 1
        # else:
        #     if_fire = -1
        
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        
        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] # 更新导弹1是否存在
        if self.n_missile1_state == True:
            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
                                #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用
        
        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)
        
        States = np.concatenate((Pos_Diff, Plane_Euler, target_angle, locked, missile1_state, Oppo_Euler, oppo_hea), axis=None)

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array([plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2]])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array([plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2]])

    def save_parameters_to_txt(self, log_dir):
        # os.makedirs(log_dir)
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)


        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, 'w') as file:
            file.write(source_code1)
            file.write(' ')
            file.write(source_code2)
            file.write(' ')
            file.write(source_code3)

    # for expert data

    def get_loc_diff(self, state):
        loc_diff = ((((state[0]) * 10000) ** 2) + (((state[1]) * 10000) ** 2) + (((state[2]) * 10000) ** 2)) ** (1 / 2)
        return loc_diff

    def get_reward(self, state, action, n_state):
        reward = 0
        step_success = 0
        loc_diff = self.get_loc_diff(n_state)  # get location difference information for reward
        
        # 距离惩罚：帮助追击
        reward -= (0.0001 * loc_diff)

        # 目标角惩罚：帮助锁敌
        reward -= (n_state[6]) * 10

        # 开火奖励：帮助开火
        if action[-1] > 0: # 如果导弹发射
            reward -= 8
            if state[8] > 0 and state[7] < 0: # 且导弹存在、不锁敌
                # reward -= 100 # 4
                step_success = -1
            elif state[8] > 0 and state[7] > 0: # 且导弹存在、锁敌
                # reward += 8 # 100
                step_success = 1
            else:
                # reward -= 10 # 1
                reward -= 0

        if n_state[-1] < 0.1:
            reward += 600

        return reward, step_success

    def get_termination(self, state):
        done = False
        if state[-1] <= 0.1: # 敌机血量低于0则结束
            done = True
        return done 

class HarfangSerpentineEnv(HarfangEnv):
    def __init__(self):
        super(HarfangSerpentineEnv, self).__init__()

    def set_ennemy_yaw(self):
        self.serpentine_step += 1

        if self.serpentine_step % self.duration == 0:
            self.serpentine_step = 0
            # 切换偏航方向（正负交替）
            self.oppo_yaw = 0.1 * (-1 if self.oppo_yaw > 0 else 1)
            self.duration = 500 # 300
        
        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(self.oppo_yaw))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()
        
        # self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0) #
            self.now_missile_state = True # 此时导弹发射
            # print("fire")
        else:
            self.now_missile_state = False
        
        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2 # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2 # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.oppo_yaw = 0
        self.serpentine_step = 0
        # 初始偏航角
        self.oppo_yaw = -0.1
        self.duration = 250 # 150

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2) # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_oppo)

class HarfangCircularEnv(HarfangEnv):
    def __init__(self):
        super(HarfangCircularEnv, self).__init__()

    def set_ennemy_yaw(self):
        self.circular_step += 1

        if self.circular_step < 100:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.02))
            df.set_plane_roll(self.Plane_ID_oppo, float(0.84))
        else:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.01))
        df.set_plane_roll(self.Plane_ID_oppo, float(0.28))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        # if self.circular_step < 1000:
        #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.02))
        #     df.set_plane_roll(self.Plane_ID_ally, float(0.81))
        # else:
        #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.01))
        # df.set_plane_roll(self.Plane_ID_ally, float(0.27))
        # df.set_plane_yaw(self.Plane_ID_ally, float(0))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()

        if float(action_ally[3] > 0): # 大于0发射导弹
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True # 此时导弹发射
        else:
            self.now_missile_state = False
        
        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0+random.randint(-100, 100), 3500+random.randint(-100, 100), -4000+random.randint(-100, 100), 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.circular_step = 0

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2) # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)

class HarfangSerpentineInfiniteEnv(HarfangSerpentineEnv):
    def __init__(self):
        super(HarfangSerpentineInfiniteEnv, self).__init__()
        self.infinite_total_success = 0
        self.infinite_total_fire = 0
        self.infinite_total_step = 0

    def step_test(self, action):
        self.infinite_total_step += 1
        if self.infinite_total_step % 60 == 0:
            df.rearm_machine(self.Plane_ID_ally)
        # df.set_health("ennemy_2", 1) # 恢复敌机健康值

        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success

    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        self._get_loc_diff()  # get location difference information for reward
        
        # 距离惩罚：帮助追击
        self.reward -= (0.0001 * (self.loc_diff)) # 0.4

        # 目标角惩罚：帮助锁敌
        self.reward -= (self.target_angle)*10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4

        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        # 开火奖励：帮助开火
        if self.now_missile_state == True: # 如果此step导弹发射
            self.reward -= 8
            if self.missile1_state == True and self.Ally_target_locked == False: # 且导弹存在、不锁敌
                # self.reward -= 100 # 4、4
                self.infinite_total_fire += 1
                self.success = -1
                print('failed to fire')
            elif self.missile1_state == True and self.Ally_target_locked == True: # 且导弹存在且锁敌
                # self.reward += 8 # 100、4
                print('successful to fire')
                self.infinite_total_fire += 1
                self.infinite_total_success += 1
                self.success = 1
                self.fire_success = True
            else:
                # self.reward -= 10
                self.reward -= 0

        # 坠落奖励（最终奖励）
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600 # 无、无
            print('enemy have fallen')