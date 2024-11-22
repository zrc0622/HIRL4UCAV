import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory

from .model import TwinnedQNetwork, GaussianPolicy
from .utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, lr, stateDim, nActions, full1Dim, full2Dim, layerNorm, name):
        super(Actor,self).__init__()
        
        self.layerNorm = layerNorm
        
        self.full1 = nn.Linear(stateDim,full1Dim)
        nn.init.kaiming_uniform_(self.full1.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm1 = nn.LayerNorm(full1Dim)
        
        self.full2 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full2.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm2 = nn.LayerNorm(full2Dim)
        
        self.final = nn.Linear(full2Dim,nActions)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,x):
        if self.layerNorm:
            
            x = F.leaky_relu(self.layernorm1(self.full1(x)))
            x = F.leaky_relu(self.layernorm2(self.full2(x)))

        else:
            
            x = F.leaky_relu(self.full1(x))
            x = F.leaky_relu(self.full2(x))

        
        return torch.tanh(self.final(x))
    
    def saveCheckpoint(self,ajan,model_name):
        torch.save(self.state_dict(), model_name + '\\{}'.format(ajan)  + self.name)
        
    def loadCheckpoint(self,ajan,model_name):
        self.load_state_dict(torch.load(model_name + '\\{}'.format(ajan) + self.name))

class SacAgent:
    def __init__(self, observation_space, action_space, log_dir, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, imitative=False, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=300, target_update_interval=3,
                 eval_interval=1000, cuda=True):
        self.observation_space = observation_space
        self.action_space = action_space

        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.bc_actor = Actor(0.0001, 14, 4, 256, 512, True, "Harfang_GYM")

        self.policy = GaussianPolicy(
            self.observation_space.shape[0],
            self.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.observation_space.shape[0],
            self.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.observation_space.shape[0],
            self.action_space.shape[0],
            hidden_units=hidden_units).to(self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.observation_space.shape,
                self.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MultiStepMemory(
                memory_size, self.observation_space.shape,
                self.action_space.shape, self.device, gamma, multi_step)
        self.expert_memory = None

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.plot_dir = os.path.join(log_dir, 'plot')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.imitative = imitative
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def load_bc_actor(self, ajan, model_name):
        self.bc_actor.loadCheckpoint(ajan, model_name) # agent20 model/bc
        self.bc_actor.eval()  # 设置为评估模式

    # 反复执行train_episode()进行训练，直到网络更新次数够
    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    # 判断是否需要更新网络
    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    # 执行一个动作
    def act(self, state):
        if self.start_steps > self.steps:
            action = self.action_space.sample()
        else:
            action = self.explore(state)
        return action

    # 带有随机性的探索
    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    # 不带随机性的探索
    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    # 训练一个回合
    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done # masked_done表示到达episode最大steps

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update(): # 如果可以更新网络，则更新网络
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0: # 验证
                self.evaluate()
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    # 更新网络
    def learn(self, if_expert, expert_num=None, expert_data=None):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            if if_expert:
               
                batch1 = self.memory.sample(self.batch_size - expert_num)

                concatenated_tensors = []
                for i in range(len(batch1)):
                    concatenated = torch.cat((batch1[i], expert_data[i]), dim=0)
                    concatenated_tensors.append(concatenated)
                # 将拼接后的张量重新组成一个元组
                batch = tuple(concatenated_tensors)
                # print('expert num is: ', expert_num)
                # print('batch is: ', batch[0].shape[0])
                
            else:
                batch = self.memory.sample(self.batch_size)
                # print('batch is: ', batch[0].shape[0])

            # set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights)
        # policy_loss, entropies = self.calc_policy_loss(batch, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        
        if self.imitative:
            policy_loss, entropies, bc_loss, bc_weight = self.calc_policy_loss(batch, weights)
        else:
            policy_loss, entropies = self.calc_policy_loss(batch, weights)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(), self.steps)

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)
            if self.imitative:
                self.writer.add_scalar(
                    'loss/bc', bc_loss.detach().item(),
                    self.learning_steps)
                self.writer.add_scalar(
                    'loss/bc_weight', bc_weight,
                    self.learning_steps)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        if self.imitative:
            with torch.no_grad():
                bc_action = self.bc_actor(states)
            bc_q1, bc_q2 = self.critic(states, bc_action)
            bc_q = torch.min(bc_q1, bc_q2)
            bc_weight = (bc_q > q).float().mean().detach().item()

            expert_batch = self.expert_memory.sample(self.batch_size)
            expert_states, expert_actions, expert_rewards, expert_next_states, expert_dones = expert_batch
            means, log_stds = self.policy.forward(expert_states)
            policy_actions = torch.tanh(means)

            bc_loss = F.mse_loss(expert_actions, policy_actions) * 10000

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        if self.imitative:
            policy_loss = torch.mean((- q - self.alpha * entropy) * weights)*(1-bc_weight) + bc_loss*bc_weight
            return policy_loss, entropy, bc_loss, bc_weight
        else:
            policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
            return policy_loss, entropy   

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    # def evaluate(self):
    #     episodes = 10
    #     returns = np.zeros((episodes,), dtype=np.float32)

    #     for i in range(episodes):
    #         state = self.env.reset()
    #         episode_reward = 0.
    #         done = False
    #         while not done:
    #             action = self.exploit(state)
    #             next_state, reward, done, _ = self.env.step(action)
    #             episode_reward += reward
    #             state = next_state
    #         returns[i] = episode_reward

    #     mean_return = np.mean(returns)

    #     self.writer.add_scalar(
    #         'reward/test', mean_return, self.steps)
    #     print('-' * 60)
    #     print(f'Num steps: {self.steps:<5}  '
    #           f'reward: {mean_return:<5.1f}')
    #     print('-' * 60)

    def save_models(self, model_name):
        self.policy.save(os.path.join(self.model_dir, f'policy_{model_name}.pth'))
        self.critic.save(os.path.join(self.model_dir, f'critic_{model_name}.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, f'critic_target_{model_name}.pth'))

    # def __del__(self):
    #     self.writer.close()
    #     self.env.close()
