import numpy as np
import dogfight_client as df
from Constants import *
import gym
import random

class DemoEnv():
    def __init__(self):
        self.done = False
        self.loc_diff = 0
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 4.0]), # gai 1发导弹0 2发导弹1 3发导弹2
                                           dtype=np.float64)
        self.Plane_ID_oppo = "ennemy_2"  # Opponent aircrafts name
        self.Plane_ID_ally = "ally_1"  # our aircrafts name
        self.Aircraft_Loc = None
        self.Ally_target_locked = None
        self.n_Ally_target_locked = False
        self.reward = 0
        self.Plane_Irtifa = 0
        self.plane_heading = 0
        self.plane_heading_2 = 0
        self.now_missile_state = False # gai 导弹此时是否发射（本次step是否发射了导弹）
        self.missile1_state = True # gai 导弹1是否存在
        self.n_missile1_state = True
        self.missile2_state = True # gai 导弹1是否存在
        self.n_missile2_state = True
        self.missile3_state = True # gai 导弹1是否存在
        self.n_missile3_state = True
        self.missile4_state = True # gai 导弹1是否存在
        self.n_missile4_state = True
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally) # gai 导弹列表
        self.missile1_id = self.missile[0]
        self.oppo_health = 1 # gai health_level

    def reset(self):  # reset simulation beginning of episode
        self.done = False
        state_ally = self._get_observation()  # get observations
        self._reset_machine()
        self._reset_missile() # gai 重设导弹
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # set target, for firing missile

        return state_ally

    def step(self, action_ally):
        self._apply_action(action_ally)  # apply neural networks output
        state_ally = self._get_observation()  # in each step, get observation
        self._get_reward()  # get reward value
        self._get_termination()  # check termination conditions

        return state_ally, self.reward, self.done, {}

    def _get_reward(self):
        self.reward = 0
        self._get_loc_diff()  # get location difference information for reward
        self.reward -= (0.0001 * (self.loc_diff))

        # if self.loc_diff < 500:
        #     self.reward += 1000

        if self.plane_heading > 180:
            deger_1 = (self.plane_heading - 360)
        else:
            deger_1 = self.plane_heading

        if self.plane_heading_2 > 180:
            deger_2 = (self.plane_heading_2 - 360)
        else:
            deger_2 = self.plane_heading_2
        self.reward -= abs(deger_1 - deger_2) / 90

        if self.Plane_Irtifa < 2000:
            self.reward -= 4

        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        # if self.now_missile_state == True: # gai 如果本次step导弹发射
        #     if self.missile1_state == False and self.Ally_target_locked == False: # 且导弹不存在、不锁敌
        #         self.reward -= 10 # 则扣10分
        #     elif self.missile1_state == True and self.Ally_target_locked == True: # 且导弹存在且锁敌
        #         self.reward += 100 # 则加1000分
        #     else:
        #         self.reward -= 5 # 则扣5分

        if self.oppo_health <= 0:
            self.reward += 200

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))
        
        self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True:
                df.fire_missile(self.Plane_ID_ally, 0) # gai
                self.now_missile_state = True # 此时导弹发射
        
        df.update_scene()
        

    def _get_termination(self):
        # if self.loc_diff < 300:
        #     self.done = True
        # if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
        #     self.done = True
        if self.oppo_health['health_level'] <= 0: # gai
            self.done = True
            return 1
        else:
            return 0

    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 1)
        self.oppo_health = 1 # gai
        x = random.randint(-1000, 1000)
        y = random.randint(2000, 4000)
        z = random.randint(1000, 3000)
        df.reset_machine_matrix(self.Plane_ID_oppo, x, y, z, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 20, -80, 0, 0, 0)
        df.update_scene()
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 250)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        # df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self): # gai
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally) # 重新装填导弹

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) + (
                    (self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) + (
                                     (self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** (1 / 2)

    def _get_observation(self):
        # Plane States
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [plane_state["position"][0] / NormStates["Plane_position"],
                     plane_state["position"][1] / NormStates["Plane_position"],
                     plane_state["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Plane_Heading = plane_state["heading"] / NormStates["Plane_heading"]

        # Opponent States
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)

        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        self.plane_heading_2 = Oppo_state["heading"]
        self.plane_heading = plane_state["heading"] # gai
        self.Plane_Irtifa = plane_state["position"][1]
        self.Aircraft_Loc = plane_state["position"]
        self.Oppo_Loc = Oppo_state["position"]
        
        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = plane_state["target_locked"]
        if self.n_Ally_target_locked == True: # gai 
            locked = 1
        else:
            locked = -1

        target_angle = plane_state['target_angle'] / 360
        Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], Plane_Pos[1] - Oppo_Pos[1], Plane_Pos[2] - Oppo_Pos[2]]
        
        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        oppo_hea = self.oppo_health['health_level'] # gai 敌机初始血量（health）为20，health_level为0.2，导弹伤害为30

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

        self.missile2_state = self.n_missile2_state
        self.n_missile2_state = Missile_state["missiles_slots"][1] # 更新导弹1是否存在
        if self.n_missile2_state == True:
            missile2_state = 1
        else:
            missile2_state = -1

        self.missile3_state = self.n_missile3_state
        self.n_missile3_state = Missile_state["missiles_slots"][2] # 更新导弹1是否存在
        if self.n_missile3_state == True:
            missile3_state = 1
        else:
            missile3_state = -1

        self.missile4_state = self.n_missile4_state
        self.n_missile4_state = Missile_state["missiles_slots"][3] # 更新导弹1是否存在
        if self.n_missile4_state == True:
            missile4_state = 1
        else:
            missile4_state = -1

        States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
                                 Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, 
                                 missile1_state, missile2_state, missile3_state, missile4_state), axis=None) # gai 感觉加入敌机健康值没用 
        
        # self.now_missile_state = False

        return States
    
    def _get_health(self):
        state = df.get_plane_state(self.Plane_ID_oppo)
        health = state['health_level']
        return health

    def _get_action(self):
        action = [0,0,0,-1]
        state = df.get_plane_state(self.Plane_ID_ally)
        # print(state)
        # {'timestamp': 70310, 'timestep': 0.016666666666666666, 'position': [-16.567607879638672, 149.64927673339844, 183.3050537109375], 'Euler_angles': [-0.7854030728340149, 0.0050728581845760345, -0.12579230964183807], 'easy_steering': True, 'health_level': 1, 'destroyed': False, 'wreck': False, 'crashed': False, 'active': True, 'type': 'AICRAFT', 'nationality': 1, 'thrust_level': 1, 'brake_level': 0, 'flaps_level': 0, 'horizontal_speed': 112.246337890625, 'vertical_speed': 97.5827865600586, 'linear_speed': 148.73345947265625, 'move_vector': [-11.339219093322754, 97.5827865600586, 111.672119140625], 'linear_acceleration': 1.4310089111328068, 'altitude': 149.64927673339844, 'heading': 0.29141239305266703, 'pitch_attitude': 45.00029076022247, 'roll_attitude': -7.207452358089359, 'post_combustion': True, 'user_pitch_level': 2.4729337383178063e-05, 'user_roll_level': -0.06377019733190536, 'user_yaw_level': 0.0, 'gear': False, 'ia': True, 'autopilot': True, 'autopilot_heading': 1.1798513347878505, 'autopilot_speed': -1, 'autopilot_altitude': 1500, 'target_id': 'ennemy_2', 'target_locked': False, 'target_out_of_range': True, 'target_angle': 6.8759194555908145}
        Pitch_Att = state["user_pitch_level"]
        Roll_Att = state["user_roll_level"]
        Yaw_Att = state["user_yaw_level"]
        action[0] = Pitch_Att
        action[1] = Roll_Att
        action[2] = Yaw_Att
        if self.missile1_state == True and self.n_missile1_state == False:
            action[3] = 1
        if self.missile2_state == True and self.n_missile2_state == False:
            action[3] = 2
        if self.missile3_state == True and self.n_missile3_state == False:
            action[3] = 3
        if self.missile4_state == True and self.n_missile4_state == False:
            action[3] = 4
        # print(action)
        return action

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array([plane_state["position"][0] / NormStates["Plane_position"],
                plane_state["position"][1] / NormStates["Plane_position"],
                plane_state["position"][2] / NormStates["Plane_position"]])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array([plane_state["position"][0] / NormStates["Plane_position"],
                plane_state["position"][1] / NormStates["Plane_position"],
                plane_state["position"][2] / NormStates["Plane_position"]])
