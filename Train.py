#IMPORTS
from NeuralNetwork import Agent
from ReplayMemory  import *
import numpy as np
import time
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from HarfangEnv_GYM import *
import dogfight_client as df

import datetime
import os
from pathlib import Path

from plot import draw_dif, draw_pos

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

print(torch.cuda.is_available())

df.connect("10.243.58.131", 50888) #TODO:Change IP and PORT values

start = time.time() #STARTING TIME
df.disable_log()

# PARAMETERS
trainingEpisodes = 6000
validationEpisodes = 100
explorationEpisodes = 200

Test = False
if Test:
    render = False
else:
    render = True
    
df.set_renderless_mode(render)
df.set_client_update_mode(True)

bufferSize = (10**6)
gamma = 0.99
criticLR = 1e-3
actorLR = 1e-3
tau = 0.005
checkpointRate = 100
highScore = -math.inf
successRate = -math.inf
batchSize = 128
maxStep = 6000
validatStep = 15000
hiddenLayer1 = 256
hiddenLayer2 = 512
stateDim = 14 # gai
actionDim = 4 # gai
useLayerNorm = True

name = "Harfang_GYM"


#INITIALIZATION
env = HarfangEnv()

agent = Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)

start_time = datetime.datetime.now()
dir = Path.cwd() # 获取工作区路径
log_dir = str(dir) + "\\" + "new_runs\\" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute) # tensorboard文件夹路径
plot_dir = log_dir + "\\" + "plot"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validatStep=validatStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2)
env.save_parameters_to_txt(log_dir)

writer = SummaryWriter(log_dir)
arttir = 1
# agent.loadCheckpoints(f"Agent{arttir-1}_") # 使用未添加导弹的结果进行训练

if not Test:
    # RANDOM EXPLORATION
    print("Exploration Started")
    for episode in range(explorationEpisodes):
        state = env.reset()
        done = False
        for step in range(maxStep):
            if not done:
                action = env.action_space.sample()                

                n_state,reward,done, info = env.step(action)
                # print(n_state)
                if step is maxStep-1:
                    done = True
                agent.store(state,action,n_state,reward,done)
                state=n_state

                if done:
                    break
        sys.stdout.write("\rExploration Completed: %.2f%%" % ((episode+1)/explorationEpisodes*100))
    sys.stdout.write("\n")

    print("Training Started")
    scores = []
    for episode in range(trainingEpisodes):
        state = env.reset()
        totalReward = 0
        done = False
        for step in range(maxStep):
            if not done:
                action = agent.chooseAction(state)
                n_state,reward,done, info = env.step(action)

                if step is maxStep - 1:
                    done = True

                agent.store(state, action, n_state, reward, done) # n_state 为下一个状态
                state = n_state
                totalReward += reward

                if agent.buffer.fullEnough(agent.batchSize):
                    critic_loss, actor_loss = agent.learn()
                    writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                    writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                    
                if done:
                    break
               
        scores.append(totalReward)
        writer.add_scalar('Training/Episode Reward', totalReward, episode)
        writer.add_scalar('Training/Last 100 Average Reward', np.mean(scores[-100:]), episode)
        
        
        now = time.time()
        seconds = int((now - start) % 60)
        minutes = int(((now - start) // 60) % 60)
        hours = int((now - start) // 3600)
        print('Episode: ', episode+1, ' Completed: %r' % done,\
            ' FinalReward: %.2f' % totalReward, \
            ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
            'RunTime: ', hours, ':',minutes,':', seconds)
            
        #VALIDATION
        success = 0
        if (((episode + 1) % checkpointRate) == 0):
            valScores = []
            dif = []
            self_pos = []
            oppo_pos = []
            for e in range(validationEpisodes):
                state = env.reset()
                totalReward = 0
                done = False
                for step in range(validatStep):
                    if not done:
                        action = agent.chooseActionNoNoise(state)
                        n_state, reward, done, info = env.step(action)
                        if step is maxStep - 1:
                            done = True

                        state = n_state
                        totalReward += reward

                        if e == validationEpisodes - 1:
                            dif.append(env.loc_diff)
                            self_pos.append(env.get_pos())
                            oppo_pos.append(env.get_oppo_pos())
                    if done:
                        if env.loc_diff < 300: # 改
                            success += 1
                        break

                valScores.append(totalReward)

            if mean(valScores) > highScore or success/validationEpisodes > successRate:
                if mean(valScores) > highScore: # 总奖励分数
                    highScore = mean(valScores)
                    agent.saveCheckpoints("Agent{}_score{}".format(arttir, highScore))
                    draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                    draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir) 

                elif success / validationEpisodes > successRate: # 追逐成功率
                    successRate = success / validationEpisodes
                    agent.saveCheckpoints("Agent{}_successRate{}".format(arttir, successRate))
                    draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                    draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir)
        
            arttir += 1

            print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', successRate)
            writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
            writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
else:
    success = 0
    for e in range(validationEpisodes):
        state = env.reset()
        totalReward = 0
        done = False
        for step in range(validatStep):
            if not done:
                action = agent.chooseActionNoNoise(state)
                n_state,reward,done, info  = env.step(action)
                if step is validatStep - 1:
                    done = True

                state = n_state
                totalReward += reward
            if done:
                if env.loc_diff < 300:
                    success += 1
                break

        # print('Test  Reward:', totalReward)
    print('Success Ratio:', success / validationEpisodes)
