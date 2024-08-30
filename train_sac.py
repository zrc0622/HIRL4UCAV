import torch
import argparse
import dogfight_client as df
import time
from agent.SAC.agent import SacAgent as SACAgent
import math
from environment.HarfangEnv_GYM import *
import gym
from pathlib import Path
import datetime
import csv
from utils.plot import draw_dif, draw_pos, plot_dif, plot_dif2, draw_pos2
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory

def validate(validationEpisodes, env:HarfangEnv, validationStep, agent:SACAgent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate):          
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
                action = agent.exploit(state)
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
            agent.save_models("Agent{}_{}_{}".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))))
            if plot:
                draw_pos(f'pos1_{arttir}.png', self_pos, oppo_pos, fire, lock, plot_dir) 
                # draw_pos2(f'pos2_{arttir}.png', self_pos, oppo_pos, plot_dir)
                # plot_dif(dif, lock, fire, plot_dir, f'my_dif1_{arttir}.png')
                plot_dif2(dif, lock, missile, fire, plot_dir, f'my_dif2_{arttir}.png')

        elif success / validationEpisodes >= successRate or arttir%10 == 0: # 追逐成功率
            successRate = success / validationEpisodes
            agent.save_models("Agent{}_{}_{}".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))))
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

    port = config.port
    render = not (config.render)
    model_name = config.model_name
    sac_type = config.type

    df.connect("10.243.58.131", port)

    start = time.time()
    df.disable_log()

    trainingEpisodes = 6000
    validationEpisodes = 20
    explorationEpisodes = 200

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
    state_space = gym.spaces.Box(low=np.array([-1.0] * stateDim), high=np.array([1.0] * stateDim), dtype=np.float64)
    action_space = gym.spaces.Box(low=np.array([-1.0] * actionDim), high=np.array([1.0] * actionDim), dtype=np.float64)
    useLayerNorm = True

    bc_actor_dir = './models/BC/bc_1'
    bc_actor_name = 'Agent20_successRate0.64'

    data_dir = './expert_data/expert_data_bc1.csv'

    name = 'Harfang_GYM'

    start_time = datetime.datetime.now()
    dir = Path.cwd()
    log_dir = str(dir) + "/" + "logs/" + "SAC/" + model_name + "/" + "log/" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=log_dir, batch_size=batchSize, lr=actorLR, hidden_units = [hiddenLayer1, hiddenLayer2],
                     memory_size=bufferSize, gamma=gamma, tau=tau)

    env = HarfangEnv()
    writer = agent.writer

    save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=agent.model_dir,data_dir=data_dir)
    env.save_parameters_to_txt(log_dir)

    arttir = 1

    if sac_type == 'sac':
        print(f'agent is SAC')

        print("Exploration Started")
        for episode in range(explorationEpisodes):
            state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()

                    n_state, reward, done, info, stepsuccess = env.step(action)

                    agent.memory.append(state, action, reward, n_state, done, done)
                    state = n_state

                    if step is maxStep-1:
                        done = True

                    if done:
                        break

        print('Training Started')
        scores = []
        trainsuccess = []
        for episode in range(trainingEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            fire = False

            for step in range(maxStep):
                if not done:
                    action = agent.explore(state)
                    n_state, reward, done, info, stepsuccess = env.step(action)

                    agent.memory.append(state, action, reward, n_state, done, done)

                    if step is maxStep - 1:
                        break

                    state = n_state
                    totalReward += reward

                    if len(agent.memory) > agent.batch_size:
                        agent.learn(if_expert = False)
                
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
            
            # VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, True, agent.plot_dir, arttir, agent.model_dir, episode, checkpointRate, writer, highScore, successRate)
                arttir += 1
    else:
        # 初始化专家缓冲区
        print(f'agent is E-SAC')
        expert_num = batchSize
        # expert_num = 0

        expert_memory = MultiStepMemory(100000, state_space.shape, action_space.shape, device, gamma, 1)
        agent.load_bc_actor(bc_actor_name, bc_actor_dir)
        
        print('initialize the expert buffer')
        for expert_episode in range(100):
            state = env.reset()
            done = False
            success = False
            temp_storage = []
            point = 0
            for step in range(maxStep):
                if not done:
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float).to(device)
                    action = agent.bc_actor(state_tensor)

                    n_state, reward, done, info, stepsuccess = env.step(action)

                    # if point == 0: # important
                    if True:
                        if action.cpu().detach().numpy()[-1] > 0:
                            success_action = action.cpu().detach().numpy()
                            success_action[-1] = 1
                            temp_storage.append((state, success_action, reward, n_state, done, done))
                        else:
                            temp_storage.append((state, action.cpu().detach().numpy(), reward, n_state, done, done))

                    if stepsuccess:
                        success = True
                        point = 1
                        
                    state = n_state

                    if step == maxStep-1:
                        done = True

                    if done:
                        break
            if success:
                for data in temp_storage:
                    expert_memory.append(*data)
                print('episode: ', expert_episode, 'step: ', step)
                # break
        print("expert buffer size is: ", len(expert_memory))

        # # test
        # for _ in range(1000):
        #     expert_data = expert_memory.sample(expert_num)
        #     agent.learn(True, expert_num, expert_data)
        
        # # test
        # batch1 = expert_memory.sample(64)
        # for item in batch1:
        #     # 打印元素及其对应的类型
        #     print(f"1{item.size()}")
        # batch2 = expert_memory.sample(32)
        # for item in batch2:
        #     # 打印元素及其对应的类型
        #     print(f"2{item.size()}")
        # concatenated_tensors = []
        # for i in range(len(batch1)):
        #     concatenated = torch.cat((batch1[i], batch2[i]), dim=0)
        #     concatenated_tensors.append(concatenated)
        # batch = tuple(concatenated_tensors)
        # for item in batch:
        #     # 打印元素及其对应的类型
        #     print(f"{item.size()}")

        print("Exploration Started")
        for episode in range(10):
            state = env.reset()
            done = False
            for step in range(maxStep):
                if not done:
                    action = env.action_space.sample()

                    n_state, reward, done, info, stepsuccess = env.step(action)

                    agent.memory.append(state, action, reward, n_state, done, done)

                    # if len(agent.memory) > agent.batch_size:
                    #     expert_data = expert_memory.sample(expert_num)
                    #     agent.learn(True, expert_num, expert_data)
                    #     if len(agent.memory)%1000 == 0 and expert_num != 0: 
                    #         expert_num -= 1

                    state = n_state

                    if step == maxStep-1:
                        done = True

                    if done:
                        break

        print('Training Started')
        scores = []
        trainsuccess = []
        for episode in range(trainingEpisodes):
            # if episode % 6 == 0 and expert_num != 0: # important
            #     expert_num -= 1
            print('expert num is: ', expert_num)
           
            state = env.reset()
            totalReward = 0
            done = False
            fire = False

            for step in range(maxStep):
                if not done:
                    action = agent.explore(state)
                    n_state, reward, done, info, stepsuccess = env.step(action)

                    agent.memory.append(state, action, reward, n_state, done, done)
       
                    if step == maxStep - 1:
                        break

                    state = n_state
                    totalReward += reward

                    if step % 10 == 0 and expert_num != 0: # important
                        expert_num -= 1

                    if len(agent.memory) > agent.batch_size:
                        expert_data = expert_memory.sample(expert_num)
                        agent.learn(True, expert_num, expert_data)
                
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
            
            # VALIDATION
            if (((episode + 1) % checkpointRate) == 0):
                highScore, successRate = validate(validationEpisodes, env, validationStep, agent, True, agent.plot_dir, arttir, agent.model_dir, episode, checkpointRate, writer, highScore, successRate)
                arttir += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--type', type=str, default='sac')
    main(parser.parse_args())