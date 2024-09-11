#IMPORTS
from agents.TD3 import Agent as TD3Agent
from agents.HIRL import Agent as HIRLAgent
from agents.BC import Agent as BCAgent
from utils.plot import draw_dif, draw_pos, plot_dif, plot_dif2, draw_pos2
from utils.data_processor import read_data
from utils.buffer import *
import numpy as np
import time
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from environments.HarfangEnv_GYM import *
import environments.dogfight_client as df
import datetime
import os
from pathlib import Path
import csv
import argparse
from tqdm import tqdm
import random

def validate(validationEpisodes, env:HarfangEnv, validationStep, agent:HIRLAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate):          
    success = 0
    valScores = []
    self_pos = []
    oppo_pos = []
    for e in range(validationEpisodes):
        dif=[]
        fire=[]
        lock=[]
        missile=[]
        state = env.reset()
        totalReward = 0
        done = False
        for step in range(validationStep):
            if not done:
                action = agent.chooseActionNoNoise(state)
                n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
                state = n_state
                totalReward += reward
                
                dif.append(env.loc_diff)
                if iffire:
                    fire.append(step)
                if locked:
                    lock.append(step)
                if beforeaction:
                    missile.append(step)
                
                if e == validationEpisodes - 1:
                    self_pos.append(env.get_pos())
                    oppo_pos.append(env.get_oppo_pos())

                if step is validationStep - 1:
                    break

            elif done:
                if env.episode_success:
                    if plot:
                        # plot_dif(dif, lock, fire, plot_dir, f'my_sdif1_{arttir}.png')
                        plot_dif2(dif, lock, missile, fire, plot_dir, f'my_sdif2_{arttir}.png')
                    success += 1
                break

        valScores.append(totalReward)
    
    if plot:
        os.makedirs(plot_dir+'/csv', exist_ok=True)
        with open(plot_dir+'/csv/self_pos{}.csv'.format(arttir), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(self_pos)
        with open(plot_dir+'/csv/oppo_pos{}.csv'.format(arttir), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(oppo_pos)
        with open(plot_dir+'/csv/fire{}.csv'.format(arttir), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows([[item] for item in fire])
        with open(plot_dir+'/csv/lock{}.csv'.format(arttir), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows([[item] for item in lock])
        with open(plot_dir+'/csv/dif{}.csv'.format(arttir), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows([[item] for item in dif])

    if mean(valScores) > highScore or success/validationEpisodes >= successRate or arttir%10 == 0:
        if mean(valScores) > highScore: # 总奖励分数
            highScore = mean(valScores)
            agent.saveCheckpoints("Agent{}_{}_{}".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
            if plot:
                draw_pos(f'pos1_{arttir}.png', self_pos, oppo_pos, fire, lock, plot_dir) 
                # draw_pos2(f'pos2_{arttir}.png', self_pos, oppo_pos, plot_dir)
                # plot_dif(dif, lock, fire, plot_dir, f'my_dif1_{arttir}.png')
                plot_dif2(dif, lock, missile, fire, plot_dir, f'my_dif2_{arttir}.png')

        elif success / validationEpisodes >= successRate or arttir%10 == 0: # 追逐成功率
            successRate = success / validationEpisodes
            agent.saveCheckpoints("Agent{}_{}_{}".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
            if plot:
                draw_pos(f'pos1_{arttir}.png', self_pos, oppo_pos, fire, lock, plot_dir)
                # draw_pos2(f'pos2_{arttir}.png', self_pos, oppo_pos, plot_dir)
                # plot_dif(dif, lock, fire, plot_dir, f'my_dif1_{arttir}.png')
                plot_dif2(dif, lock, missile, fire, plot_dir, f'my_dif2_{arttir}.png')

    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes)
    tensor_writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
    tensor_writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
    return highScore, successRate

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

def main(config):
    print('gpu is ' + str(torch.cuda.is_available()))

    agent_name = config.agent
    model_name = config.model_name
    port = config.port
    hirl_type = config.type
    bc_weight = config.bc_weight
    test_mode = config.test_mode
    render = not (config.render)
    plot = config.plot

    if config.seed is not None:
        random.seed(config.seed)

    if not render:
        print('rendering mode')
    else:
        print('no rendering mode')

    df.connect("10.241.58.131", port) #TODO:Change IP and PORT values

    start = time.time() #STARTING TIME
    df.disable_log()

    # PARAMETERS
    trainingEpisodes = 6000
    validationEpisodes = 20 # 20
    explorationEpisodes = 200 # 200

    # # Test = True
    # if Test:
    #     render = True
    # else:
    #     render = True
        
    df.set_renderless_mode(render)
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
    useLayerNorm = True

    bc_actor_dir = '/models/BC/bc_1'
    bc_actor_name = 'Agent20_successRate0.64'

    data_dir = './expert_data/expert_data_bc1.csv'
    data_folder_dir = './expert_data'
    expert_states, expert_actions = read_data(data_dir)

    name = "Harfang_GYM"

    if agent_name == 'HIRL':
        agent = HIRLAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight)
    elif agent_name == 'BC':
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)
    elif agent_name == 'TD3':
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)
    
    log_dir = ''
    model_dir = log_dir + '/model'
    model_name = '' # Agent24_100_427
    
    agent.loadCheckpoints(model_name, model_dir)

    if test_mode == 1:
        print('test mode 1')
        env = RandomHarfangEnv()
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
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
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
        env = InfiniteHarfangEnv()
        success = 0
        validationEpisodes = 10
        for e in range(validationEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            validationStep = 3000
            for step in range(validationStep):
                if not done:
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action, step)
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
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
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
        env = RandomHarfangEnv()
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
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
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
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
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
            
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='HIRL') # 代理：HIRL、TD3
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--type', type=str, default='linear') # HIRL type：linear、fixed、soft
    parser.add_argument('--bc_weight', type=float, default=1)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--test_mode', type=int, default=1) # 1为随机初始化模式，2为无限导弹模式
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    main(parser.parse_args())