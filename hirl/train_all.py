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
import time
import math
from statistics import mean, pstdev
import datetime
import os
import csv
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

def validate(validationEpisodes, env:HarfangEnv, validationStep, agent:HIRLAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate, if_random):          
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
        
    if mean(valScores) > highScore or success/validationEpisodes >= successRate or arttir % 5 == 0:
        agent.saveCheckpoints("Agent{}_{}_{}_".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
        if plot:
            plot_3d_trajectories(self_pos, oppo_pos, fire, lock, plot_dir, f'trajectories_{arttir}.png') 
            plot_distance(distance, lock, missile, fire, plot_dir, f'distance_{arttir}.png')
            
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
            with open(plot_dir+'/csv/distance{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows([[item] for item in distance])

        if mean(valScores) > highScore: # 总奖励分数
            highScore = mean(valScores)
        if success / validationEpisodes >= successRate: # 追逐成功率
            successRate = success / validationEpisodes

    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes, ' Fire Success Rate:', fire_success/validationEpisodes)
    tensor_writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
    tensor_writer.add_scalar('Validation/Std Reward', pstdev(valScores), episode)
    tensor_writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
    tensor_writer.add_scalar('Validation/Fire Success Rate', fire_success/validationEpisodes, episode)
    
    if success / validationEpisodes >= 0.5 and not if_random:
        random_validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer, highScore, successRate)

    return highScore, successRate

def random_validate(validationEpisodes, env:HarfangEnv, validationStep, agent:HIRLAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate):          
    t_fire_success = []
    t_valScores = []
    for _ in range(5):
        success = 0
        fire_success = 0
        valScores = []
        for _ in range(validationEpisodes):
            state = env.random_reset()
            totalReward = 0
            done = False
            for step in range(validationStep):
                if not done:
                    action = agent.chooseActionNoNoise(state)
                    n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
                    state = n_state
                    totalReward += reward

                    if step == validationStep - 1:
                        break

                elif done:
                    if env.episode_success:
                        success += 1
                    if env.fire_success:
                        fire_success += 1
                    break

            valScores.append(totalReward)
        t_valScores.append(mean(valScores))
        t_fire_success.append(fire_success/validationEpisodes)

    tensor_writer.add_scalar('Random_Validation/Mean Reward', mean(t_valScores), episode)
    tensor_writer.add_scalar('Random_Validation/Std Reward', pstdev(t_valScores), episode)
    tensor_writer.add_scalar('Random_Validation/Mean Fire Success Rate', mean(t_fire_success), episode)
    tensor_writer.add_scalar('Random_Validation/Std Fire Success Rate', pstdev(t_fire_success), episode)

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

    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update the 'network.ip' field in config.yaml with your own IP address.")

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
    batchSize = 128 # 128
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 13
    actionDim = 4
    useLayerNorm = True
    expert_warm_up = True
    warm_up_rate = 10
    bc_warm_up = False
    bc_warm_up_weight = 0 # 不能动

    if if_random: data_dir = local_config['experiment']['expert_data_dir'] + f'/{env_type}/expert_data_ai_random.csv'
    elif not if_random: data_dir = f'hirl/data/{env_type}/expert_data_ai_fixed_small_delta0.csv'

    start_time = datetime.datetime.now()
    log_dir = local_config["experiment"]["result_dir"] + "/" + env_type + "/" + agent_name + "/" + model_name + "/" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)
    model_dir = os.path.join(log_dir, 'model')
    summary_dir = os.path.join(log_dir, 'summary')
    plot_dir = os.path.join(log_dir, 'plot')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

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

    save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=model_dir,hirl_type=hirl_type, data_dir=data_dir)
    env.save_parameters_to_txt(log_dir)

    writer = SummaryWriter(summary_dir)
    
    arttir = 1
    if load_model:
        agent.loadCheckpoints(f"Agent20_successRate0.64", model_dir)

    if agent_name == 'BC':
        print('agent is BC')
        
        for episode in range(6000):
            for step in range(maxStep):
                bc_loss = agent.train_actor()
                if step == 1000:
                    writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
            now = time.time()
            seconds = int((now - start) % 60)
            minutes = int(((now - start) // 60) % 60)
            hours = int((now - start) // 3600)
            print('Episode: ', episode+1, 'RunTime: ', hours, ':',minutes,':', seconds)

            # validation
            if ((episode + 1) % checkpointRate == 0 and (episode + 1) >= 1000):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
                arttir += 1
    
    elif agent_name == 'HIRL':
        print(f'agent is HIRL, HIRL type is {hirl_type}')

        # RANDOM EXPLORATION
        print("Exploration Started")
        for episode in range(explorationEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()                
                    n_state, reward, done, info, step_success = env.step(action)
                    
                    if step == maxStep-1:
                        break
                        # reward -= 200
                        # done = True
                        
                    agent.store(state,action,n_state,reward,done,step_success)
                    
                    state=n_state

        print("Training Started")
        
        expert_num = batchSize
        if expert_warm_up:
            print("initialize the expert buffer")
            i = 0
            count = 0
            while i < expert_states.shape[0]:
                state = expert_states[i]
                action = expert_actions[i]
                next_state = expert_states[i+1]
                reward, step_success = env.get_reward(state, action, next_state)
                if reward > 400: print("right")
                done = env.get_termination(next_state)
                
                if done: 
                    i += 1
                    count += 1
                    print(count)
                
                i += 1

                agent.expert_buffer.store(state, action, next_state, reward, done, step_success)
            
            print(f"expert buffer size is {len(agent.expert_buffer.memory)}")


        if hirl_type == 'soft':
            agent.load_bc_actor(bc_actor_name, bc_actor_dir)
            print('success load bc model')
            # agent.actor.loadCheckpoint(bc_actor_name, bc_actor_dir)
            # agent.targetActor.loadCheckpoint(bc_actor_name, bc_actor_dir)
            # print('success load pretrained bc model')
        scores = []
        trainsuccess = []
        firesuccess = []
        for episode in range(trainingEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            totalReward = 0
            done = False
            shut_down = False
            fire = False

            if hirl_type == 'linear':
                bc_weight_now = bc_weight - episode/5000
                if bc_weight_now <= 0:
                    bc_weight_now = 0
            elif hirl_type == 'fixed':
                bc_weight_now = bc_weight
            elif hirl_type == 'soft':
                if bc_warm_up:
                    bc_warm_up_weight = 0.3 - episode/1000
                    if bc_warm_up_weight <= 0:
                        bc_warm_up_weight = 0
                bc_weight_now = 100    

            for step in range(maxStep):
                if not done:
                    action = agent.chooseAction(state)
                    n_state, reward, done, info, step_success = env.step(action)

                    if step == maxStep - 1:
                        break
                        # reward -= 200
                        # done = True

                    agent.store(state, action, n_state, reward, done, step_success)
                    
                    state = n_state
                    totalReward += reward

                    if step % warm_up_rate == 0 and expert_num != 0: # important
                        expert_num -= 1
                        # print(f"expert num is {expert_num}")

                    if agent.buffer.fullEnough(agent.batchSize):
                        critic_loss, actor_loss, bc_loss, rl_loss, bc_fire_loss, bc_weight_now = agent.learn(bc_weight_now, expert_num, bc_warm_up_weight)
                        if step % logRate == 0:
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
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % shut_down, ' Fire Success: ', fire, \
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds, 'BC_weight: ', bc_weight_now) # completed表示回合结束，fire表示成功击落或撞落
                
            #VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
                arttir += 1
    
    elif agent_name == 'TD3':
        print('agent is TD3')

        # RANDOM EXPLORATION
        print("Exploration Started")
        for episode in range(explorationEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()                
                    n_state, reward, done, info, step_success = env.step(action)
                    
                    if step == maxStep-1:
                        break

                    agent.store(state,action,n_state,reward,done,step_success)
                    
                    state=n_state

        print("Training Started")
        scores = []
        trainsuccess = []
        firesuccess = []
        for episode in range(trainingEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            totalReward = 0
            done = False
            shut_down = False # 表示是否成功
            fire = False
            for step in range(maxStep):
                if not done:
                    action = agent.chooseAction(state)
                    n_state, reward, done, info , step_success= env.step(action)

                    if step == maxStep - 1:
                        break

                    agent.store(state, action, n_state, reward, done, step_success) # n_state 为下一个状态
                    
                    state = n_state
                    totalReward += reward

                    if agent.buffer.fullEnough(agent.batchSize):
                        critic_loss, actor_loss = agent.learn()
                        if step % logRate == 0:
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
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % shut_down, ' Fire Success: ', fire,\
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds)
                
            #VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
                arttir += 1       
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='HIRL', 
                        help="Specify the agent to use: 'HIRL', 'BC' or 'TD3'. Default is 'HIRL'.")
    parser.add_argument('--port', type=int, default=None,
                        help="The port number for the training environment. Example: 12345.")
    parser.add_argument('--type', type=str, default='soft',
                        help="Type of HIRL algorithm to use: 'linear', 'fixed', or 'soft'. Default is 'soft'.")
    parser.add_argument('--bc_weight', type=float, default=0.5,
                        help="Weight for the behavior cloning (BC) loss. Default is 0.5.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Name of the model to be saved. Example: 'HIRL_soft'.")
    parser.add_argument('--load_model', action='store_true',
                        help="Flag to load a pre-trained model. Use this if you want to resume training.")
    parser.add_argument('--render', action='store_true',
                        help="Flag to enable rendering of the environment.")
    parser.add_argument('--plot', action='store_true',
                        help="Flag to plot training metrics. Use this for visualization.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility. Default is None (no seed).")
    parser.add_argument('--env', type=str, default='straight_line',
                        help="Specify the training environment type: 'straight_line', 'serpentine' or 'circular'. Default is 'straight_line'.")
    parser.add_argument('--random', action='store_true',
                        help="Flag to use random initialization in the environment. Default is False.")
    main(parser.parse_args())