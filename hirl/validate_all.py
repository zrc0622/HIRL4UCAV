from agents.TD3 import Agent as TD3Agent
from agents.HIRL import Agent as HIRLAgent
from agents.BC import Agent as BCAgent
from utils.plot import plot_3d_trajectories, plot_distance
from utils.data_processor import read_data
from utils.buffer import *
from utils.seed import *
from environments.HarfangEnv_GYM import *
import environments.dogfight_client as df

import numpy as np
import statistics
import time
from tqdm import tqdm
import math
from statistics import mean, pstdev
import datetime
import os
import csv
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

def validate(validationEpisodes, env:HarfangSerpentineInfiniteEnv, validationStep, agent:HIRLAgent, if_random=True, if_infinite=True):
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
                action = agent.chooseActionNoNoise(state)
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

    if if_infinite:
        return mean(valScores), fire_success/validationEpisodes, env.infinite_total_success/env.infinite_total_fire
    else:
        return mean(valScores), fire_success/validationEpisodes, 0

def main(config):
    print('gpu is ' + str(torch.cuda.is_available()))

    agent_name = config.agent
    model_name = config.model_name
    port = config.port
    hirl_type = config.type
    bc_weight = config.bc_weight
    load_model = config.load_model
    render = not (config.render)
    plot = config.plot
    env_type = config.env
    
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

    start = time.time() #STARTING TIME
    df.disable_log()

    name = "Harfang_GYM"

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

    elif env_type == "circular":
        print("env is harfang circular")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1900 # 6000
        validationStep = 1900 # 6000
            
        env = HarfangCircularEnv()

    df.set_renderless_mode(render)
    df.set_client_update_mode(True)

    bufferSize = 10**5 # 10**6
    gamma = 0.99
    criticLR = 1e-3
    actorLR = 1e-3
    tau = 0.005
    checkpointRate = 25 # 25
    logRate = 300
    highScore = -math.inf
    successRate = 0
    batchSize = 128 #128
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 13
    actionDim = 4
    useLayerNorm = True
    expert_warm_up = True
    warm_up_rate = 10

    if if_random: data_dir = local_config['experiment']['expert_data_dir'] + f'/{env_type}/expert_data_ai_random.csv'
    elif not if_random: data_dir = local_config['experiment']['expert_data_dir'] + f'/{env_type}/expert_data_ai.csv'

    if agent_name == 'HIRL':
        expert_states, expert_actions = read_data(data_dir)
        bc_actor_dir = local_config['experiment']['bc_actor_dir'] + f'/{env_type}'
        bc_actor_name = local_config['experiment']['bc_actor_name'][f'{env_type}'][:-len(name)]
        agent = HIRLAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight, expert_warm_up)
    elif agent_name == 'BC':
        expert_states, expert_actions = read_data(data_dir)
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)
    elif agent_name == 'TD3':
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)

    model_dir = 'B:/code/HIRL4UCAV/TAAS_After_Revision/result/straight_line/HIRL/hirl_random_5/2024_11_12_10_41/model'
    model_name = 'Agent55_100_5_'
    agent.loadCheckpoints(model_name, model_dir)
    if_infinite = config.infinite
    if if_infinite: 
        env = HarfangSerpentineInfiniteEnv()
        validationStep = 1200
    else:
        env = HarfangSerpentineEnv()
        validationStep = 1200

    nums = 2
    return_list = []
    successrate_list = []
    fire_list = []
    for _ in tqdm(range(nums)):
        r, s, f = validate(validationEpisodes, env, validationStep, agent, if_random, if_infinite) 
        return_list.append(r)
        successrate_list.append(s)
        fire_list.append(f)
    if if_infinite:
        print(statistics.mean(fire_list), statistics.stdev(fire_list))
    else:
        print(statistics.mean(return_list), statistics.stdev(return_list))
        print(statistics.mean(successrate_list), statistics.stdev(successrate_list))       
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='HIRL') # 代理：HIRL、TD3
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--type', type=str, default='linear') # HIRL type：linear、fixed、soft
    parser.add_argument('--bc_weight', type=float, default=1)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env', type=str, default='straight_line') # serpentine
    parser.add_argument('--random', action='store_true') # serpentine
    parser.add_argument('--infinite', action='store_true')
    main(parser.parse_args())

# python hirl/validate_all.py --agent HIRL --port 54321 --type soft --env serpentine --random --infinite