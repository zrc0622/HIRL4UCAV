import hirl.environments.dogfight_client as df
from hirl.environments.constants import *
from hirl.environments.HarfangEnv_GYM import HarfangEnv, HarfangEnvNew
import random

class AIHarfangEnv(HarfangEnv):
    def __init__(self):
        super(AIHarfangEnv, self).__init__()
    
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
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.n_missile1_state = Missile_state["missiles_slots"][0]
        if self.missile1_state == True and self.n_missile1_state == False:
            action[3] = 1
        # print(action)
        return action
    
    def fire(self):
        df.fire_missile(self.Plane_ID_ally, 0)

class AIHarfangEnvNew(HarfangEnvNew):
    def __init__(self):
        super(AIHarfangEnvNew, self).__init__()
    
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
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.n_missile1_state = Missile_state["missiles_slots"][0]
        if self.missile1_state == True and self.n_missile1_state == False:
            action[3] = 1
        # print(action)
        return action
    
    def fire(self):
        df.fire_missile(self.Plane_ID_ally, 0)

class AIHarfangEnvNewRandom(AIHarfangEnvNew):
    def __init__(self):
        super(AIHarfangEnvNewRandom, self).__init__()
    
    def _reset_machine(self):
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