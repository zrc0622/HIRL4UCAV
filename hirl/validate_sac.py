import os
os.environ['kmp_duplicate_lib_ok']='true'

from agents.SAC.agent import SacAgent as SACAgent
from utils.plot import plot_3d_trajectories, plot_distance
from utils.data_processor import read_data
from utils.seed import *
from environments.HarfangEnv_GYM import *
import environments.dogfight_client as df
from rltorch.memory import MultiStepMemory

import torch
import gym
import time
from tqdm import tqdm
import math
from statistics import mean, pstdev
import datetime
import os
import csv
import argparse
import yaml
import statistics
from torch.utils.tensorboard import SummaryWriter

def validate(validationEpisodes, env:HarfangSerpentineInfinite, validationStep, agent, if_random=True):
    env.infinite_total_success = 0
    env.infinite_total_fire = 0
    success = 0
    fire_success = 0
    valScores = []
    self_pos = []
    oppo_pos = []
    for e in range(validationEpisodes):
        distance=[]
        fire=[]
        lock=[]
        missile=[]
        if if_random: state = env.random_reset()
        else: state = env.reset()
        totalReward = 0
        done = False
        for step in range(validationStep):
            if not done:
                action = agent.exploit(state)
                n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
                state = n_state
                totalReward += reward
                
                distance.append(env.loc_diff)
                if iffire:
                    fire.append(step)
                if locked:
                    lock.append(step)
                if beforeaction:
                    missile.append(step)
                
                if e == validationEpisodes - 1:
                    self_pos.append(env.get_pos())
                    oppo_pos.append(env.get_oppo_pos())

                if step == validationStep - 1:
                    break

            elif done:
                if env.episode_success:
                    success += 1
                if env.fire_success:
                    fire_success += 1
                break

        valScores.append(totalReward)
    
    return mean(valScores), fire_success/validationEpisodes, env.infinite_total_success/env.infinite_total_fire

def main(config):
    print('gpu is ' + str(torch.cuda.is_available()))

    port = config.port
    render = not (config.render)
    model_name = config.model_name
    sac_type = config.type
    env_type = config.env
    plot = config.plot

    if config.random:
        print("random")
    else:
        print("fixed")
    if_random = config.random

    if config.seed is not None:
        set_seed(config.seed)
        print(f"successful set seed: {config.seed}")
    else:
        print("no seed is set")

    if not render:
        print('rendering mode')
    else:
        print('no rendering mode')

    with open('local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)

    df.connect(local_config["network"]["ip"], port)

    start = time.time()
    df.disable_log()

    # PARAMETERS
    if env_type == "straight_line":
        print("env is harfang straight line")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1500 # 6000
        validationStep = 1500 # 6000

        env = HarfangEnv()

    elif env_type == "serpentine":
        print("env is harfang serpentine")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1500 # 6000
        validationStep = 1500 # 6000

        env = HarfangSerpentineEnv()

    df.set_renderless_mode(render)
    df.set_client_update_mode(True)

    bufferSize = 2*(10**5)
    gamma = 0.99
    criticLR = 1e-3
    actorLR = 1e-3
    tau = 0.005
    checkpointRate = 25 # 25
    highScore = -math.inf
    successRate = 0
    batchSize = 128 #128
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 13
    actionDim = 4
    state_space = gym.spaces.Box(low=np.array([-1.0] * stateDim), high=np.array([1.0] * stateDim), dtype=np.float64)
    action_space = gym.spaces.Box(low=np.array([-1.0] * actionDim), high=np.array([1.0] * actionDim), dtype=np.float64)
    useLayerNorm = True
    warm_up_rate = 20

    if if_random: data_dir = local_config['experiment']['expert_data_dir'] + f'/{env_type}/expert_data_ai_random.csv'
    elif not if_random: data_dir = local_config['experiment']['expert_data_dir'] + f'hirl/data/{env_type}/expert_data_ai.csv'

    agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=None, batch_size=batchSize, lr=actorLR, hidden_units = [hiddenLayer1, hiddenLayer2],
                     memory_size=bufferSize, gamma=gamma, tau=tau)

    agent.policy.load('./result/serpentine/SAC/esac_random_1/log/2024_11_2_19_40/model/policy_Agent42_100_19_.pth')

    infinite = True
    if infinite: 
        env = HarfangSerpentineInfinite()
        validationStep = 1200

    nums = 2
    return_list = []
    successrate_list = []
    fire_list = []
    for _ in tqdm(range(nums)):
        r, s, f = validate(validationEpisodes, env, validationStep, agent, if_random) 
        return_list.append(r)
        successrate_list.append(s)
        fire_list.append(f)
    if not infinite:
        print(statistics.mean(return_list), statistics.stdev(return_list))
        print(statistics.mean(successrate_list), statistics.stdev(successrate_list))
    else:
        print(statistics.mean(fire_list), statistics.stdev(fire_list))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--type', type=str, default='ESAC')
    parser.add_argument('--env', type=str, default='straight_line') # serpentine
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--random', action='store_true') # serpentine
    main(parser.parse_args())