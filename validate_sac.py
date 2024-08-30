import torch
import argparse
import dogfight_client as df
import time
from agent.SAC.agent import SacAgent as SACAgent
import math
from environment.HarfangEnv_GYM import *
from environment.HarfangEnv_GYM_test1 import *
from environment.HarfangEnv_GYM_test2 import *
import gym
from pathlib import Path
import datetime
import csv
from utils.plot import draw_dif, draw_pos, plot_dif, plot_dif2, draw_pos2
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory
import random
from tqdm import tqdm

def main(config):
    test_mode = config.test_mode
    if config.seed is not None:
        random.seed(config.seed)

        print('gpu is ' + str(torch.cuda.is_available()))

    port = config.port


    df.connect("10.243.58.131", port)

    start = time.time()
    df.disable_log()

    trainingEpisodes = 6000
    validationEpisodes = 20
    explorationEpisodes = 200

    df.set_renderless_mode(True)
    df.set_client_update_mode(True)

    bufferSize = (10**6)
    gamma = 0.99
    criticLR = 1e-3
    actorLR = 1e-3
    tau = 0.005
    checkpointRate = 25 # 25
    highScore = -math.inf
    successRate = 0
    batchSize = 128 #128
    maxStep = 6000 # 6000
    validationStep = 6000 # 6000
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 14
    actionDim = 4
    state_space = gym.spaces.Box(low=np.array([-1.0] * stateDim), high=np.array([1.0] * stateDim), dtype=np.float64)
    action_space = gym.spaces.Box(low=np.array([-1.0] * actionDim), high=np.array([1.0] * actionDim), dtype=np.float64)
    useLayerNorm = True

    bc_actor_dir = './models/BC/bc_1'
    bc_actor_name = 'Agent20_successRate0.64'

    name = 'Harfang_GYM'

    start_time = datetime.datetime.now()
    dir = Path.cwd()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = 'logs4/SAC/esac_34/log/2024_3_2_10_17'
    model_dir = log_dir + '/model'
    model_name = 'Agent7_100_-1430.pth'

    agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=log_dir, batch_size=batchSize, lr=actorLR, hidden_units = [hiddenLayer1, hiddenLayer2],
                     memory_size=bufferSize, gamma=gamma, tau=tau)

    env = HarfangEnv()

    agent.policy.load(model_dir + '/policy_' + model_name)
    agent.critic.load(model_dir + '/critic_' + model_name)
    agent.critic_target.load(model_dir + '/critic_target_' + model_name)

    if test_mode == 1:
        print('test mode 1')
        env = HarfangEnv_test1()
        success = 0
        fire_success = 0
        validationEpisodes = 50
        r=0
        for e in tqdm(range(validationEpisodes)):
            state = env.reset()
            point = 0
            totalReward = 0
            done = False
            for step in range(validationStep):
                if not done:
                    action = agent.exploit(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, reward, step_success   = env.step_test(action)
                    state = n_state
                    totalReward += reward

                    if step is validationStep - 1:
                        break
                    
                    if done:
                        if env.episode_success:
                            success += 1
                        if point == 1:
                            fire_success += 1
                    
                    if step_success == 1:
                        point = 1
            r += totalReward
        print('Fianl reward:', r/validationEpisodes)              
        print('Fall Success Ratio:', success / validationEpisodes)
        print('Fire Success Ratio', fire_success / validationEpisodes)

    elif test_mode == 2:
        print('test mode 2')
        env = HarfangEnv_test2()
        success = 0
        validationEpisodes = 10
        for e in range(validationEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            validationStep = 3000
            for step in range(validationStep):
                if not done:
                    action = agent.exploit(state)
                    n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward, step_success = env.step_test(action, step)
                    state = n_state
                    totalReward += reward

                    if step == validationStep - 1:
                        print('total fire is:', env.total_fire,'total success is', env.total_success)
                        break

                elif done:
                    if env.episode_success:
                        print('total fire is:', env.total_fire,'total success is', env.total_success)
                        print('two aircrafts collided')
                    break
    
    elif test_mode == 3:
        print('test mode 3')
        env = HarfangEnv()
        success = 0
        fire_success = 0
        validationEpisodes = 50
        r = 0
        for e in tqdm(range(validationEpisodes)):
            state = env.reset()
            totalReward = 0
            point = 0
            done = False
            for step in range(validationStep):
                if not done:
                    action = agent.exploit(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, reward, step_success   = env.step_test(action)
                    state = n_state
                    totalReward += reward

                    if step is validationStep - 1:
                        break
                    
                    if done:
                        if env.episode_success:
                            success += 1
                        if point == 1:
                            fire_success += 1
                    
                    if step_success == 1:
                        point = 1
            r += totalReward
        print('Fianl reward:', r/validationEpisodes)           
        print('Fall Success Ratio:', success / validationEpisodes)
        print('Fire Success Ratio', fire_success / validationEpisodes)

    elif test_mode == 4:
        print('test mode 4')
        env = HarfangEnv_test1()
        success = 0
        fire_success = 0
        validationEpisodes = 50
        r = 0
        for e in tqdm(range(validationEpisodes)):
            state = env.reset()
            point = 0
            totalReward = 0
            done = False
            for step in range(validationStep):
                if not done:
                    action = agent.exploit(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, reward, step_success   = env.step_test(action)
                    state = n_state
                    totalReward += reward

                    if step is validationStep - 1:
                        break
                    
                    if done:
                        if env.episode_success:
                            success += 1
                        if point == 1:
                            fire_success += 1
                    
                    if step_success == 1:
                        point = 1
            r += totalReward
        print('Fianl reward:', r/validationEpisodes)
    elif test_mode == 5:
        print('test mode 5')
        env = HarfangEnv()
        success = 0
        fire_success = 0
        validationEpisodes = 50
        r = 0
        for e in tqdm(range(validationEpisodes)):
            state = env.reset()
            totalReward = 0
            point = 0
            done = False
            for step in range(validationStep):
                if not done:
                    action = agent.exploit(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, reward, step_success   = env.step_test(action)
                    state = n_state
                    totalReward += reward

                    if step is validationStep - 1:
                        break
                    
                    if done:
                        if env.episode_success:
                            success += 1
                        if point == 1:
                            fire_success += 1
                    
                    if step_success == 1:
                        point = 1
            r += totalReward
        print('Fianl reward:', r/validationEpisodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test_mode', type=int, default=1)
    parser.add_argument('--port', type=int, default=11111)
    main(parser.parse_args())

# python validate_sac.py --test_mode 1 --port 11111 --seed 1 