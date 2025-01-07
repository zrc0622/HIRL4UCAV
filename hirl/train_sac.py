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
import math
from statistics import mean, pstdev
import datetime
import csv
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter


def validate(validationEpisodes, env:HarfangEnv, validationStep, agent:SACAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate, if_random):          
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
    
    if mean(valScores) > highScore or success/validationEpisodes >= successRate or arttir % 5 == 0:
        agent.save_models("Agent{}_{}_{}_".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))))
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
    
    return highScore, successRate

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

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

    if config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update the 'network.ip' field in config.yaml with your own IP address.")

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

    start_time = datetime.datetime.now()

    log_dir = local_config["experiment"]["result_dir"] + "/" + env_type + "/" + "SAC/" + model_name + "/" + "log/" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=log_dir, batch_size=batchSize, lr=actorLR, hidden_units = [hiddenLayer1, hiddenLayer2],
                     memory_size=bufferSize, gamma=gamma, tau=tau)

    writer = agent.writer

    save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=agent.model_dir,data_dir=data_dir)
    env.save_parameters_to_txt(log_dir)

    arttir = 1

    if sac_type == 'SAC':
        print(f'agent is SAC')

        print("Exploration Started")
        for episode in range(explorationEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()

                    n_state, reward, done, info, stepsuccess = env.step(action)

                    if step == maxStep-1:
                        break
                        # done = True

                    agent.memory.append(state, action, reward, n_state, done, done)
                    state = n_state



        print('Training Started')
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

            for step in range(maxStep):
                if not done:
                    action = agent.explore(state)
                    n_state, reward, done, info, stepsuccess = env.step(action)

                    if step == maxStep - 1:
                        break

                    agent.memory.append(state, action, reward, n_state, done, done)

                    state = n_state
                    totalReward += reward

                    if len(agent.memory) > agent.batch_size:
                        agent.learn(if_expert = False)
                
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
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, ' Fire Success: %r' % fire,\
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds)
            
            # VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, agent.plot_dir, arttir, agent.model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
                arttir += 1
    else:
        # 初始化专家缓冲区
        print(f'agent is E-SAC')
        expert_num = batchSize
        expert_states, expert_actions = read_data(data_dir)

        agent.expert_memory = MultiStepMemory(len(expert_states) + 10, state_space.shape, action_space.shape, device, gamma, 1)
        print('initialize the expert buffer')

        print("load ai expert data:")

        i = 0
        count = 0
        while i < expert_states.shape[0]:
            state = expert_states[i]
            action = expert_actions[i]
            next_state = expert_states[i+1]
            reward, step_success = env.get_reward(state, action, next_state)   
            if reward > 400: 
                print(f'hit: {i}')
            done = env.get_termination(next_state)
            
            if done: 
                print(f'done: {i}')
                i += 1
                count += 1
                print(count)
            
            i += 1

            agent.expert_memory.append(state, action, reward, next_state, done, done)
        
        print("expert buffer size is: ", len(agent.expert_memory))

        print("Exploration Started")
        for episode in range(explorationEpisodes):
            if if_random: state = env.random_reset()
            else: state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()

                    n_state, reward, done, info, stepsuccess = env.step(action)
                 
                    if step == maxStep-1:
                        break
                        # reward -= 200
                        # done = True

                    agent.memory.append(state, action, reward, n_state, done, done)

                    # if len(agent.memory) > agent.batch_size:
                    #     expert_data = expert_memory.sample(expert_num)
                    #     agent.learn(True, expert_num, expert_data)
                    #     if len(agent.memory)%1000 == 0 and expert_num != 0: 
                    #         expert_num -= 1

                    state = n_state

        print('Training Started')
        scores = []
        trainsuccess = []
        firesuccess = []
        for episode in range(trainingEpisodes):
            # if episode % 6 == 0 and expert_num != 0: # important
            #     expert_num -= 1
           
            if if_random: state = env.random_reset()
            else: state = env.reset()
            totalReward = 0
            done = False
            shut_down = False
            fire = False

            for step in range(maxStep):
                if not done:
                    action = agent.explore(state)
                    n_state, reward, done, info, stepsuccess = env.step(action)

                    if step == maxStep - 1:
                        break
                        # reward -= 200
                        # done = True

                    agent.memory.append(state, action, reward, n_state, done, done)

                    state = n_state
                    totalReward += reward

                    if step % warm_up_rate == 0 and expert_num != 0:
                        expert_num -= 1

                    if len(agent.memory) > agent.batch_size:
                        expert_data = agent.expert_memory.sample(expert_num)
                        agent.learn(True, expert_num, expert_data)
                
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
            print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, ' Fire Success: %r' % fire,\
                ' FinalReward: %.2f' % totalReward, \
                ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                'RunTime: ', hours, ':',minutes,':', seconds)
            
            # VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, agent.plot_dir, arttir, agent.model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
                arttir += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None,
                        help="The port number for the training environment. Example: 12345.")
    parser.add_argument('--render', action='store_true',
                        help="Flag to enable rendering of the environment.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Name of the model to be saved. Example: 'E-SAC'.")
    parser.add_argument('--type', type=str, default='ESAC',
                        help="Type of SAC algorithm to use: 'SAC' or 'ESAC'. Default is 'ESAC'.")
    parser.add_argument('--env', type=str, default='straight_line',
                        help="Specify the training environment type: 'straight_line', 'serpentine' or 'circular'. Default is 'straight_line'.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility. Default is None (no seed).")
    parser.add_argument('--plot', action='store_true',
                        help="Flag to plot training metrics. Use this for visualization.")
    parser.add_argument('--random', action='store_true',
                        help="Flag to use random initialization in the environment. Default is False.")
    main(parser.parse_args())