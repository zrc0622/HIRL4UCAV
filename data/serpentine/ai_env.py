import numpy as np
import environments.dogfight_client as df
from environments.constants import *
import gym
import random

from environments.HarfangEnv_GYM import HarfangSerpentineEnv
from data.straight_line.ai_env import AIHarfangEnv

class AIHarfangSerpentineEnv(HarfangSerpentineEnv, AIHarfangEnv):
    def __init__(self):
        super(AIHarfangSerpentineEnv, self).__init__()