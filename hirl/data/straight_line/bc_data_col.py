import torch
import argparse
import environments.dogfight_client as df
import time
from agents.SAC.agent import SacAgent as SACAgent
import math
from environments.HarfangEnv_GYM import *
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
                n_state,reward,done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
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

def main(config):
    print('gpu is ' + str(torch.cuda.is_available()))

    port = config.port
    render = not (config.render)
    model_name = config.model_name
    sac_type = config.type

    df.connect("10.241.58.131", port)

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

    bc_actor_dir = 'models\\BC\\bc_1'
    bc_actor_name = 'Agent20_successRate0.64'

    name = 'Harfang_GYM'

    start_time = datetime.datetime.now()
    dir = Path.cwd()
    log_dir = str(dir) + "\\" + "logs4\\" + "SAC\\" + model_name + "\\" + "log\\" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=log_dir, batch_size=batchSize, lr=actorLR, hidden_units = [hiddenLayer1, hiddenLayer2],
                     memory_size=bufferSize, gamma=gamma, tau=tau)

    env = HarfangEnv()
    writer = agent.writer

    arttir = 1


    # 初始化专家缓冲区
    print(f'agent is E-SAC')
    expert_num = batchSize
    # expert_num = 0

    expert_memory = MultiStepMemory(100000, state_space.shape, action_space.shape, device, gamma, 1)
    agent.load_bc_actor(bc_actor_name, bc_actor_dir)
    
    state_list = []
    action_list = []

    print('initialize the expert buffer')
    for expert_episode in range(16):
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
                    print('1:', step)
                    point = 1
                    
                state = n_state

                if step == maxStep-1:
                    done = True

                if done:
                    break
        if success:
            print('2：', step)
            for data in temp_storage:
                temp_state, temp_action = data[:2]  # 假设状态和动作是元组的前两个元素
                state_list.append(temp_state)
                action_list.append(temp_action)
                expert_memory.append(*data)
            print('episode: ', expert_episode, 'step: ', step)
            # break
    print("expert buffer size is: ", len(expert_memory))

    filename = 'expert_data_bc1.csv'

    action_array = np.array(action_list)
    state_array = np.array(state_list)
    print(action_array.shape)
    data = [state_array, action_array]

    with open(filename, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
        writer = csv.writer(file)
        writer.writerows(data)  # 将数据写入CSV文件
    print("ok")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--type', type=str, default='sac')
    main(parser.parse_args())


# python demo_BC.py --type esac --port 12345 --model_name esac