import numpy as np
import dogfight_client as df
from Constants import *
import gym


class HarfangEnv():
    def __init__(self):
        self.done = False
        self.loc_diff = 0
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), # gai
                                           dtype=np.float64)
        self.Plane_ID_oppo = "ennemy_2"  # Opponent aircrafts name
        self.Plane_ID_ally = "ally_1"  # our aircrafts name
        self.Aircraft_Loc = None
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False # 运用动作后是否锁敌
        self.reward = 0
        self.Plane_Irtifa = 0
        self.plane_heading = 0
        self.plane_heading_2 = 0
        self.now_missile_state = False # gai 导弹此时是否发射（本次step是否发射了导弹）
        self.missile1_state = True # gai 导弹1是否存在
        self.n_missile1_state = True # gai 运用动作后导弹是否存在
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally) # gai 导弹列表
        self.missile1_id = self.missile[0]
        self.oppo_health = 0.2 # gai

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

        # if self.oppo_health <= 0:
        #     self.reward += 200

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))
        
        self.now_missile_state = False

        if float(action_ally[3] > 0): # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
                df.fire_missile(self.Plane_ID_ally, 0) # gai
                self.now_missile_state = True # 此时导弹发射
        
        df.update_scene()
        

    def _get_termination(self):
        if self.loc_diff < 300:
            self.done = True
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0: # gai 敌机血量低于0则结束
            self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1") # gai 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2) # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2 # gai
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self): # gai
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally) # 重新装填导弹

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) + (
                    (self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) + (
                                     (self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** (1 / 2)

    def _get_observation(self): # 注意get的是n_state
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
        oppo_hea = self.oppo_health['health_level'] # gai 敌机初始血量为20

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

        States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
                                 Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # gai 感觉加入敌机健康值没用
        
        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

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
