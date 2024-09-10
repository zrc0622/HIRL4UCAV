import numpy as np
import environment.dogfight_client as df
from Constants import *
import gym
import random

from environment.HarfangEnv_GYM import HarfangSerpentineEnv
from expert_data.straight_line.ai_env import AIHarfangEnv

class AIHarfangSerpentineEnv(HarfangSerpentineEnv, AIHarfangEnv):
    def __init__(self):
        super(AIHarfangSerpentineEnv, self).__init__()

    def set_ennemy_yaw(self):
        self.serpentine_step += 1

        if self.serpentine_step % self.duration == 0:
            self.serpentine_step = 0
            # 切换偏航方向（正负交替）
            self.oppo_yaw = random.uniform(0.1, 0.1) * (-1 if self.oppo_yaw > 0 else 1)
            self.duration = random.randint(300, 320)
        
        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(self.oppo_yaw))
