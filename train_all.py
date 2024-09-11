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
    fire_success = 0
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
                n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward, step_success = env.step_test(action)
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
                if env.fire_success:
                    fire_success += 1
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
            agent.saveCheckpoints("Agent{}_{}_{}_".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
            if plot:
                draw_pos(f'pos1_{arttir}.png', self_pos, oppo_pos, fire, lock, plot_dir) 
                # draw_pos2(f'pos2_{arttir}.png', self_pos, oppo_pos, plot_dir)
                # plot_dif(dif, lock, fire, plot_dir, f'my_dif1_{arttir}.png')
                plot_dif2(dif, lock, missile, fire, plot_dir, f'my_dif2_{arttir}.png')

        elif success / validationEpisodes >= successRate or arttir%10 == 0: # 追逐成功率
            successRate = success / validationEpisodes
            agent.saveCheckpoints("Agent{}_{}_{}_".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
            if plot:
                draw_pos(f'pos1_{arttir}.png', self_pos, oppo_pos, fire, lock, plot_dir)
                # draw_pos2(f'pos2_{arttir}.png', self_pos, oppo_pos, plot_dir)
                # plot_dif(dif, lock, fire, plot_dir, f'my_dif1_{arttir}.png')
                plot_dif2(dif, lock, missile, fire, plot_dir, f'my_dif2_{arttir}.png')

    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes, ' Fire Success Rate:', fire_success/validationEpisodes)
    tensor_writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
    tensor_writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
    tensor_writer.add_scalar('Validation/Fire Success Rate', fire_success/validationEpisodes, episode)
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
    load_model = config.load_model
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
    explorationEpisodes = 20 # 200

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

    #INITIALIZATION
    env = HarfangEnv()

    start_time = datetime.datetime.now()
    dir = Path.cwd() # 获取工作区路径
    log_dir = "logs/" + agent_name + "/" + model_name + "/" + "log/" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)
    model_dir = os.path.join(log_dir, 'model')
    log_dir = str(dir) + "/" + log_dir # tensorboard文件夹路径
    summary_dir = os.path.join(log_dir, 'summary')
    plot_dir = os.path.join(log_dir, 'plot')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if agent_name == 'HIRL':
        agent = HIRLAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight)
    elif agent_name == 'BC':
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)
    elif agent_name == 'TD3':
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)

    save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=model_dir,hirl_type=hirl_type, data_dir=data_dir)
    env.save_parameters_to_txt(log_dir)

    writer = SummaryWriter(summary_dir)
    
    arttir = 1
    if load_model:
        agent.loadCheckpoints(f"Agent20_successRate0.64", model_dir)

    if agent_name == 'BC':
        print('agent is BC')
        for episode in range(2000):
            for step in range(maxStep):
                bc_loss = agent.train_actor()
                writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
            now = time.time()
            seconds = int((now - start) % 60)
            minutes = int(((now - start) // 60) % 60)
            hours = int((now - start) // 3600)
            print('Episode: ', episode+1, 'RunTime: ', hours, ':',minutes,':', seconds)

            # validation
            if (((episode + 1) % 25) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, 100, writer, highScore, successRate)
                arttir += 1
    
    elif agent_name == 'HIRL':
        print(f'agent is HIRL, HIRL type is {hirl_type}')
        # RANDOM EXPLORATION
        print("Exploration Started")
        for episode in range(explorationEpisodes):
            state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()                

                    n_state,reward,done, info, stepsuccess = env.step(action)
                    # print(n_state)
                    if step is maxStep-1:
                        done = True
                    agent.store(state,action,n_state,reward,done,stepsuccess)
                    state=n_state

                    if done:
                        break

        print("Training Started")
        if hirl_type == 'soft':
            agent.load_bc_actor(bc_actor_name, bc_actor_dir)
            print('success load bc model')
        scores = []
        trainsuccess = []
        firesuccess = []
        for episode in range(trainingEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            shut_down = False
            fire = False

            if hirl_type == 'linear':
                bc_weight_now = bc_weight - episode/1000
                if bc_weight_now <= 0:
                    bc_weight_now = 0
            elif hirl_type == 'fixed':
                bc_weight_now = bc_weight
            elif hirl_type == 'soft':
                # if episode < 100:
                #     bc_weight_now = 0.9
                # else:
                #     bc_weight_now = 100
                bc_weight_now = 100    

            for step in range(maxStep):
                if not done:
                    action = agent.chooseAction(state)
                    n_state,reward,done, info, stepsuccess = env.step(action)

                    if step is maxStep - 1:
                        break

                    agent.store(state, action, n_state, reward, done, stepsuccess) # n_state 为下一个状态
                    
                    state = n_state
                    totalReward += reward

                    if agent.buffer.fullEnough(agent.batchSize):
                        critic_loss, actor_loss, bc_loss, rl_loss, bc_fire_loss, bc_weight_now = agent.learn(bc_weight_now)
                        writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                        writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                        writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
                        writer.add_scalar('Loss/RL_Loss', rl_loss, step + episode * maxStep)     
                        writer.add_scalar('Loss/BC_Fire_Loss', bc_fire_loss, step + episode * maxStep)
                        
                elif done: # done可能是过高、过低、击落、撞落
                    if env.episode_success: shut_down = True # fire可能是击落、撞落
                    if env.fire_success: fire = True
                    break

            scores.append(totalReward)
            if shut_down:
                trainsuccess.append(1)
            else:
                trainsuccess.append(0)
            if fire:
                firesuccess.append(1)
            else:
                firesuccess.append(0)
            writer.add_scalar('Training/Episode Reward', totalReward, episode)
            writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
            writer.add_scalar('Training/Average Step Reward', totalReward/step, episode)
            writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
            writer.add_scalar('Training/Last 50 Episode Fire success rate', np.mean(firesuccess[-50:]), episode)
            writer.add_scalar('Others/BC_weight', bc_weight_now, episode)               
            
            now = time.time()
            seconds = int((now - start) % 60)
            minutes = int(((now - start) // 60) % 60)
            hours = int((now - start) // 3600)
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % shut_down, ' Fire Success: %r' % fire,\
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds, 'BC_weight: ', bc_weight_now) # completed表示回合结束，fire表示成功击落或撞落
                
            #VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate)
                arttir += 1
    
    elif agent_name == 'TD3':
        print('agent is TD3')
        # RANDOM EXPLORATION
        print("Exploration Started")
        for episode in range(explorationEpisodes):
            state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()                

                    n_state,reward,done, info, stepsuccess = env.step(action)
                    # print(n_state)
                    if step is maxStep-1:
                        done = True
                    agent.store(state,action,n_state,reward,done,stepsuccess)
                    state=n_state

                    if done:
                        break

        print("Training Started")
        scores = []
        trainsuccess = []
        firesuccess = []
        for episode in range(trainingEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            shut_down = False # 表示是否成功
            fire = False
            for step in range(maxStep):
                if not done:
                    action = agent.chooseAction(state)
                    n_state,reward,done, info , stepsuccess= env.step(action)

                    if step is maxStep - 1:
                        break

                    agent.store(state, action, n_state, reward, done, stepsuccess) # n_state 为下一个状态
                    state = n_state
                    totalReward += reward

                    if agent.buffer.fullEnough(agent.batchSize):
                        critic_loss, actor_loss = agent.learn()
                        writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                        writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                        
                elif done:
                    if env.episode_success: shut_down = True
                    if env.fire_success: fire = True
                    break
                
            scores.append(totalReward)
            if shut_down:
                trainsuccess.append(1)
            else:
                trainsuccess.append(0)
            if fire:
                firesuccess.append(1)
            else:
                firesuccess.append(0)
            writer.add_scalar('Training/Episode Reward', totalReward, episode)
            writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
            writer.add_scalar('Training/Average Step Reward', totalReward/step, episode)
            writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
            writer.add_scalar('Training/Last 50 Episode Fire success rate', np.mean(firesuccess[-50:]), episode)
            
            now = time.time()
            seconds = int((now - start) % 60)
            minutes = int(((now - start) // 60) % 60)
            hours = int((now - start) // 3600)
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % shut_down, ' Fire Success: %r' % fire,\
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds)
                
            #VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate)
                arttir += 1       
    
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
    main(parser.parse_args())