#IMPORTS
from Network.TD3 import Agent as TD3Agent
from Network.ROT import Agent as ROTAgent
from Network.BC import Agent as BCAgent
from Tools.plot import draw_dif, draw_pos, plot_dif, plot_dif2, draw_pos2
from Tools.read_data import read_data
from ReplayMemory import *
import numpy as np
import time
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from Environment.HarfangEnv_GYM import *
from Environment.HarfangEnv_GYM_test1 import *
from Environment.HarfangEnv_GYM_test2 import *
import dogfight_client as df
import datetime
import os
from pathlib import Path
import csv
import argparse
from tqdm import tqdm
import random

def validate(validationEpisodes, env:HarfangEnv, validationStep, agent:ROTAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate):          
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
    rot_type = config.type
    bc_weight = config.bc_weight
    Test = config.test
    test_mode = config.test_mode
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

    bc_actor_dir = 'models\\BC\\bc_1'
    bc_actor_name = 'Agent20_successRate0.64'

    data_dir = './expert_data/expert_data_bc1.csv'
    data_folder_dir = './expert_data'
    expert_states, expert_actions = read_data(data_dir)

    name = "Harfang_GYM"

    if agent_name == 'ROT':
        agent = ROTAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight)
        model_dir = 'models/ROT/' + model_name
    elif agent_name == 'BC':
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)
        model_dir = 'models/BC/' + model_name
    elif agent_name == 'TD3':
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)
        model_dir = 'models/TD3/' + model_name

    if not Test:
        #INITIALIZATION
        env = HarfangEnv()

        start_time = datetime.datetime.now()
        dir = Path.cwd() # 获取工作区路径
        log_dir = str(dir) + "\\" + "logs5\\" + agent_name + "\\" + model_name + "\\" + "log\\" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute) # tensorboard文件夹路径
        plot_dir = log_dir + "\\" + "plot"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=model_dir,rot_type=rot_type, data_dir = data_dir)
        env.save_parameters_to_txt(log_dir)

        writer = SummaryWriter(log_dir)
    
    arttir = 1
    if load_model or Test:
        agent.loadCheckpoints(f"Agent20_successRate0.64", model_dir)

    if not Test:
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
        
        elif agent_name == 'ROT':
            print(f'agent is ROT, ROT type is {rot_type}')
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
            if rot_type == 'soft':
                agent.load_bc_actor(bc_actor_name, bc_actor_dir)
                print('success load bc model')
            scores = []
            trainsuccess = []
            for episode in range(trainingEpisodes):
                state = env.reset()
                totalReward = 0
                done = False
                fire = False

                if rot_type == 'linear':
                    bc_weight_now = bc_weight - episode/1000
                    if bc_weight_now <= 0:
                        bc_weight_now = 0
                elif rot_type == 'fixed':
                    bc_weight_now = bc_weight
                elif rot_type == 'soft':
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
                        if env.episode_success:
                            fire = True # fire可能是击落、撞落
                        break
                scores.append(totalReward)
                if fire:
                    trainsuccess.append(1)
                else:
                    trainsuccess.append(0)
                writer.add_scalar('Training/Episode Reward', totalReward, episode)
                writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
                writer.add_scalar('Training/Average Step Reward', totalReward/step, episode)
                writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
                writer.add_scalar('Others/BC_weight', bc_weight_now, episode)               
                
                now = time.time()
                seconds = int((now - start) % 60)
                minutes = int(((now - start) // 60) % 60)
                hours = int((now - start) // 3600)
                print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, \
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
            for episode in range(trainingEpisodes):
                state = env.reset()
                totalReward = 0
                done = False
                fire = False # 表示是否成功
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
                        if env.episode_success:
                            fire = True
                        break
                    
                scores.append(totalReward)
                if fire:
                    trainsuccess.append(1)
                else:
                    trainsuccess.append(0)
                writer.add_scalar('Training/Episode Reward', totalReward, episode)
                writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
                writer.add_scalar('Training/Average Step Reward', totalReward/step, episode)
                writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
                
                now = time.time()
                seconds = int((now - start) % 60)
                minutes = int(((now - start) // 60) % 60)
                hours = int((now - start) // 3600)
                print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, \
                    ' FinalReward: %.2f' % totalReward, \
                    ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                    'RunTime: ', hours, ':',minutes,':', seconds)
                    
                #VALIDATION
                if (((episode + 1) % checkpointRate) == 0):
                    highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate)
                    arttir += 1
    else:
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
                        action = agent.chooseActionNoNoise(state)
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
                        action = agent.chooseActionNoNoise(state)
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
                        action = agent.chooseActionNoNoise(state)
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
                        action = agent.chooseActionNoNoise(state)
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
                        action = agent.chooseActionNoNoise(state)
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
            
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ROT') # 代理：ROT、TD3
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--type', type=str, default='linear') # ROT type：linear、fixed、soft
    parser.add_argument('--bc_weight', type=float, default=1)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_mode', type=int, default=1) # 1为随机初始化模式，2为无限导弹模式
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    main(parser.parse_args())