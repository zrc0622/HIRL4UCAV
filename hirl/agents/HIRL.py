import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from hirl.utils.buffer import *

# model_name = 'new_pursue_model/rlbc2'

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Critic(nn.Module): 
    def __init__(self, lr, stateDim, nActions, full1Dim, full2Dim, layerNorm,name):
        super(Critic,self).__init__()
        
        self.layerNorm = layerNorm
        #Q1
        self.full1 = nn.Linear(stateDim+nActions, full1Dim)
        nn.init.kaiming_uniform_(self.full1.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm1 = nn.LayerNorm(full1Dim)
        
        self.full2 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full2.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm2 = nn.LayerNorm(full2Dim)
        
        self.final1 = nn.Linear(full2Dim,1)
        
        #Q2
        self.full3 = nn.Linear(stateDim+nActions, full1Dim)
        nn.init.kaiming_uniform_(self.full3.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm3 = nn.LayerNorm(full1Dim)
        
        self.full4 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full4.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm4 = nn.LayerNorm(full2Dim)
        
        self.final2 = nn.Linear(full2Dim,1)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,state,action):
        
        stateaction = torch.cat([state,action],1)
        
        if self.layerNorm:
            
            Q1 = F.relu(self.layernorm1(self.full1(stateaction)))
            Q1 = F.relu(self.layernorm2(self.full2(Q1)))        
            Q1 = self.final1(Q1)
        
            Q2 = F.relu(self.layernorm3(self.full3(stateaction)))
            Q2 = F.relu(self.layernorm4(self.full4(Q2)))
            Q2 = self.final2(Q2)

        else:
            
            Q1 = F.relu(self.full1(stateaction))
            Q1 = F.relu(self.full2(Q1))        
            Q1 = self.final1(Q1)
        
            Q2 = F.relu(self.full3(stateaction))
            Q2 = F.relu(self.full4(Q2))        
            Q2 = self.final2(Q2)

        
        return Q1, Q2
    
    def onlyQ1(self,state,action):
        
        stateaction = torch.cat([state,action],1)
        
        if self.layerNorm:
            
            Q1 = F.relu(self.layernorm1(self.full1(stateaction)))
            Q1 = F.relu(self.layernorm2(self.full2(Q1)))        
            Q1 = self.final1(Q1)

        else:
            Q1 = F.relu(self.full1(stateaction))
            Q1 = F.relu(self.full2(Q1))        
            Q1 = self.final1(Q1)

        return Q1
    
    def saveCheckpoint(self,ajan,model_name):
        torch.save(self.state_dict(), model_name + '\\{}'.format(ajan) + self.name)
        
    def loadCheckpoint(self,ajan,model_name):
        self.load_state_dict(torch.load(model_name + '\\{}'.format(ajan)  + self.name, map_location=self.device))
            
class Actor(nn.Module):
    def __init__(self, lr, stateDim, nActions, full1Dim, full2Dim, layerNorm, name):
        super(Actor,self).__init__()
        
        self.layerNorm = layerNorm
        
        self.full1 = nn.Linear(stateDim,full1Dim)
        nn.init.kaiming_uniform_(self.full1.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm1 = nn.LayerNorm(full1Dim)
        
        self.full2 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full2.weight, a= 0.01, mode='fan_in', nonlinearity='relu')
        
        self.layernorm2 = nn.LayerNorm(full2Dim)
        
        self.final = nn.Linear(full2Dim,nActions)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,x):
        if self.layerNorm:
            
            x = F.relu(self.layernorm1(self.full1(x)))
            x = F.relu(self.layernorm2(self.full2(x)))

        else:
            
            x = F.relu(self.full1(x))
            x = F.relu(self.full2(x))

        
        return torch.tanh(self.final(x))
    
    def saveCheckpoint(self,ajan,model_name):
        torch.save(self.state_dict(), model_name + '\\{}'.format(ajan)  + self.name)
        
    def loadCheckpoint(self,ajan,model_name):
        self.load_state_dict(torch.load(model_name + '\\{}'.format(ajan) + self.name, map_location=self.device))
    
class Agent(nn.Module):
    def __init__(self, actorLR, criticLR, stateDim, actionDim,full1Dim,full2Dim, tau, gamma, bufferSize, batchSize,\
                 layerNorm, name, expert_states, expert_actions, bc_weight, expert_warm_up):
        super(Agent,self).__init__()

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tau = tau
        self.gamma = gamma
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.actorTrainable = True
        self.actionDim = actionDim
        self.actionNoise = 0.1 # 0.1
        self.TD3LearningNoise = 0.2 # 0.2
        self.TD3LearningNoiseClamp = 0.5 # 0.5
        self.expert_states = expert_states
        self.expert_actions = expert_actions
        self.bc_weight = bc_weight
        self.expert_warm_up = expert_warm_up
        self.target_indices = self.upsample_expert_data(self.expert_actions)
        print(f"upsample expert data num is {np.size(self.target_indices)}")
        self.update_count = 0
        
        self.actor = Actor(actorLR,stateDim,actionDim, full1Dim, full2Dim, layerNorm, 'Actor_'+name).to(self.device)
        self.targetActor = Actor(actorLR,stateDim,actionDim, full1Dim, full2Dim, layerNorm, 'TargetActor_'+name).to(self.device)
        hard_update(self.targetActor, self.actor)
        
        self.critic = Critic(criticLR, stateDim, actionDim, full1Dim, full2Dim, layerNorm, 'Critic_'+name).to(self.device)
        self.targetCritic = Critic(criticLR, stateDim, actionDim, full1Dim, full2Dim, layerNorm, 'TargetCritic_'+name).to(self.device)
        hard_update(self.targetCritic, self.critic)

        self.bc_actor = Actor(actorLR,stateDim,actionDim, full1Dim, full2Dim, layerNorm, name).to(self.device)
        
        self.loss_lambda = 10000
        self.target_update_freq = 3
        self.buffer_upsample = False # train buffer stepsuccess upsample
        self.expert_upsample = False # bc fire upsample

        print(f"buffer upsample is {self.buffer_upsample}, expert upsample is {self.expert_upsample}, loss lambda is {self.loss_lambda}")

        self.buffer = UniformMemory(bufferSize, self.buffer_upsample)
        self.expert_buffer = UniformMemory(len(self.expert_states) + 10, False)
    
    def chooseAction(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = (self.actor(state) + \
                  (torch.normal(mean=torch.zeros(self.actionDim),std=torch.ones(self.actionDim)*self.actionNoise)).to(self.actor.device))\
            .clamp(-1,+1)
        return action.cpu().detach().numpy()
    
    def chooseActionSmallNoise(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = (self.actor(state) + \
                  (torch.normal(mean=torch.zeros(self.actionDim),std=torch.ones(self.actionDim)*self.actionNoise/10)).to(self.actor.device))\
            .clamp(-1,+1)
        return action.cpu().detach().numpy()
    
    def chooseActionNoNoise(self, state):
        self.actor.eval()
        state = torch.from_numpy(state).float().to(self.actor.device)
        action = self.actor(state)
        return action.cpu().detach().numpy()
    
    def store(self, *args):
        self.buffer.store(*args)
        
    def upsample_expert_data(self, BCActions):
        target_indices = np.where(BCActions[:, 3] == 1)[0]
        return target_indices 

    def learn(self, bc_weight_now, expert_num_now, bc_warm_up_weight=0):

        #SAMPLING
        if self.expert_warm_up and expert_num_now:
            # print(f"expert num is {expert_num_now}")
            batchState1, batchAction1, batchNextState1, batchReward1, batchDone1 = self.buffer.sample(self.batchSize - expert_num_now)
            batchState2, batchAction2, batchNextState2, batchReward2, batchDone2 = self.expert_buffer.sample(expert_num_now)

            batchState = torch.tensor(np.array(batchState1+batchState2), dtype=torch.float).to(self.critic.device)
            batchAction = torch.tensor(np.array(batchAction1+batchAction2), dtype=torch.float).to(self.critic.device)
            batchNextState = torch.tensor(np.array(batchNextState1+batchNextState2), dtype=torch.float).to(self.critic.device)
            batchReward = torch.tensor(np.array(batchReward1+batchReward2), dtype=torch.float).to(self.critic.device)
            batchDone = torch.tensor(np.array(batchDone1+batchDone2), dtype=torch.float).to(self.critic.device)
            
        else:
            if isinstance(self.buffer, UniformMemory):
                batchState, batchAction, batchNextState, batchReward, batchDone = self.buffer.sample(self.batchSize)
        
            batchState = torch.tensor(np.array(batchState), dtype=torch.float).to(self.critic.device)
            batchAction = torch.tensor(np.array(batchAction), dtype=torch.float).to(self.critic.device)
            batchNextState = torch.tensor(np.array(batchNextState), dtype=torch.float).to(self.critic.device)
            batchReward = torch.tensor(np.array(batchReward), dtype=torch.float).to(self.critic.device)
            batchDone = torch.tensor(np.array(batchDone), dtype=torch.float).to(self.critic.device)

        BCStates = torch.tensor(np.array(self.expert_states), dtype=torch.float).to(self.critic.device)
        BCActions = torch.tensor(np.array(self.expert_actions), dtype=torch.float).to(self.critic.device)

        samples_num = BCStates.shape[0]
        batch_indices = np.random.choice(samples_num, self.batchSize, replace=False)
        BCbatchState = BCStates[batch_indices]
        BCbatchAction = BCActions[batch_indices]

        if self.expert_upsample:
            selected_target_indices = np.random.choice(self.target_indices)
            replace_index = np.random.randint(self.batchSize)
            BCbatchState[replace_index] = BCStates[selected_target_indices]
            BCbatchAction[replace_index] = BCActions[selected_target_indices]
        
        self.targetActor.eval()
        self.targetCritic.eval()
        self.critic.eval()
        
        #NOISE REGULATION
        targetNextActions = self.targetActor(batchNextState)
        noise = (torch.normal(mean=torch.zeros(self.actionDim),std=torch.ones(self.actionDim)*self.TD3LearningNoise)).to(self.actor.device)
        noise = noise.clamp(-self.TD3LearningNoiseClamp,self.TD3LearningNoiseClamp)
        targetNextActions = (targetNextActions + noise).clamp(-1,+1)
        
        #TWIN TARGET CRITIC
        targetQ1, targetQ2 = self.targetCritic(batchNextState,targetNextActions)
        targetQmin = torch.min(targetQ1,targetQ2)
        
        #BELLMAN
        targetQ = batchReward.reshape(-1,1) + (self.gamma*targetQmin*((1-batchDone).reshape(-1,1))).detach()
        
        #CURRENT CRITIC
        currentQ1, currentQ2 = self.critic(batchState,batchAction)
        currentQ = torch.min(currentQ1,currentQ2)
        
    
        #CRITIC UPDATE
        self.critic.train()
        if isinstance(self.buffer,UniformMemory):
            self.critic_loss = (F.mse_loss(currentQ1,targetQ) + F.mse_loss(currentQ2,targetQ)) 
       
        self.critic.optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic.optimizer.step()

        #ACTOR UPDATE
        if self.actorTrainable is True:
            self.bc_weight = bc_weight_now

            self.actor.train()
            self.nextAction = self.actor(batchState)
            rl_Q = self.critic.onlyQ1(batchState, self.nextAction)
            self.rl_loss = -rl_Q.mean() # rl loss

            if self.bc_weight == 100:
                with torch.no_grad():
                    soft_action = self.bc_actor(batchState)
                soft_Q = self.critic.onlyQ1(batchState, soft_action)
                self.bc_weight = (soft_Q>rl_Q).float().mean().detach()
                self.bc_weight = self.bc_weight.item()
                # self.bc_weight = self.logistic_sigmoid(self.bc_weight)
                self.bc_weight += bc_warm_up_weight
            
            if self.bc_weight > 1: self.bc_weight = 1

            BCnextAction = self.actor(BCbatchState)
            self.bc_loss = F.mse_loss(BCnextAction, BCbatchAction) * self.loss_lambda

            # firebatchAction = BCbatchAction[:, 3]
            # firenextAction = BCnextAction[:, 3]
            # self.bc_fire_loss = F.mse_loss(firenextAction, firebatchAction) * self.loss_lambda

            firenextAction = BCnextAction[:, 3].detach()
            firebatchAction = BCbatchAction[:, 3].detach()
            self.bc_fire_loss = F.mse_loss(firenextAction, firebatchAction).item() * self.loss_lambda

            self.actor_loss = self.bc_loss * self.bc_weight + self.rl_loss * (1 - self.bc_weight)# / self.loss_lambda

            self.actor.optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor.optimizer.step()
        
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                soft_update(self.targetCritic, self.critic, self.tau)
                soft_update(self.targetActor, self.actor, self.tau)
        
        self.actorTrainable = not self.actorTrainable
        
        return self.critic_loss.mean().cpu().detach().numpy(), self.actor_loss.cpu().detach().numpy(), self.bc_loss.cpu().detach().numpy(), self.rl_loss.cpu().detach().numpy(), self.bc_fire_loss, self.bc_weight
        
    def saveCheckpoints(self,ajan,model_name):
        self.critic.saveCheckpoint(ajan,model_name)
        self.actor.saveCheckpoint(ajan,model_name)
        self.targetCritic.saveCheckpoint(ajan,model_name)
        self.targetActor.saveCheckpoint(ajan,model_name)
        
    def loadCheckpoints(self,ajan,model_name):
        self.critic.loadCheckpoint(ajan,model_name)
        self.actor.loadCheckpoint(ajan,model_name)
        self.targetCritic.loadCheckpoint(ajan,model_name)
        self.targetActor.loadCheckpoint(ajan,model_name)

    def load_bc_actor(self, ajan, model_name):
        self.bc_actor.loadCheckpoint(ajan, model_name)
        self.bc_actor.eval()

    def logistic_sigmoid(self, x, k=10):
        return 1 / (1 + np.exp(-k * (x - 0.5)))