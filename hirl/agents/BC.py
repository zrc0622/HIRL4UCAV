import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from utils.buffer import *

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
        nn.init.kaiming_uniform_(self.full1.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm1 = nn.LayerNorm(full1Dim)
        
        self.full2 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full2.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm2 = nn.LayerNorm(full2Dim)
        
        self.final1 = nn.Linear(full2Dim,1)
        
        #Q2
        self.full3 = nn.Linear(stateDim+nActions, full1Dim)
        nn.init.kaiming_uniform_(self.full3.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm3 = nn.LayerNorm(full1Dim)
        
        self.full4 = nn.Linear(full1Dim,full2Dim)
        nn.init.kaiming_uniform_(self.full4.weight, a= 0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.layernorm4 = nn.LayerNorm(full2Dim)
        
        self.final2 = nn.Linear(full2Dim,1)
        
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,state,action):
        
        stateaction = torch.cat([state,action],1)
        
        if self.layerNorm:
            
            Q1 = F.leaky_relu(self.layernorm1(self.full1(stateaction)))
            Q1 = F.leaky_relu(self.layernorm2(self.full2(Q1)))        
            Q1 = self.final1(Q1)
        
            Q2 = F.leaky_relu(self.layernorm3(self.full3(stateaction)))
            Q2 = F.leaky_relu(self.layernorm4(self.full4(Q2)))
            Q2 = self.final2(Q2)

        else:
            
            Q1 = F.leaky_relu(self.full1(stateaction))
            Q1 = F.leaky_relu(self.full2(Q1))        
            Q1 = self.final1(Q1)
        
            Q2 = F.leaky_relu(self.full3(stateaction))
            Q2 = F.leaky_relu(self.full4(Q2))        
            Q2 = self.final2(Q2)

        
        return Q1, Q2
    
    def onlyQ1(self,state,action):
        
        stateaction = torch.cat([state,action],1)
        
        if self.layerNorm:
            
            Q1 = F.leaky_relu(self.layernorm1(self.full1(stateaction)))
            Q1 = F.leaky_relu(self.layernorm2(self.full2(Q1)))        
            Q1 = self.final1(Q1)

        else:
            Q1 = F.leaky_relu(self.full1(stateaction))
            Q1 = F.leaky_relu(self.full2(Q1))        
            Q1 = self.final1(Q1)

        return Q1
    
    def saveCheckpoint(self,ajan,model_name):
        torch.save(self.state_dict(), model_name + '\\{}'.format(ajan) + self.name)
        
    def loadCheckpoint(self,ajan,model_name):
        self.load_state_dict(torch.load(model_name + '\\{}'.format(ajan)  + self.name))
            
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

class Agent:
    def __init__(self, actorLR, stateDim, actionDim, full1Dim, full2Dim, layerNorm, name, batchsize, expert_states, expert_actions):
        self.actor = Actor(actorLR, stateDim, actionDim, full1Dim, full2Dim, layerNorm, name)
        self.batchsize = batchsize
        self.expert_states = expert_states
        self.expert_actions = expert_actions
        self.target_indices = self.upsample_expert_data(self.expert_actions)

        self.expert_upsample = False

    def upsample_expert_data(self, BCActions):
        target_indices = np.where(BCActions[:, 3] == 1)[0]
        return target_indices 

    def train_actor(self):
        self.actor.train()

        BCStates = torch.tensor(np.array(self.expert_states), dtype=torch.float).to(self.actor.device)
        BCActions = torch.tensor(np.array(self.expert_actions), dtype=torch.float).to(self.actor.device)

        samples_num = BCStates.shape[0]
        batch_indices = np.random.choice(samples_num, self.batchsize, replace=False)
        BCbatchState = BCStates[batch_indices]
        BCbatchAction = BCActions[batch_indices]

        if self.expert_upsample:
            print(f"upsample num is {len(self.target_indices)}")
            selected_target_indices = np.random.choice(self.target_indices)
            replace_index = np.random.randint(self.batchsize)
            BCbatchState[replace_index] = BCStates[selected_target_indices]
            BCbatchAction[replace_index] = BCActions[selected_target_indices]

        BCnextAction = self.actor(BCbatchState)
        self.bc_loss = F.mse_loss(BCnextAction, BCbatchAction)

        self.actor.optimizer.zero_grad()
        self.bc_loss.backward()
        self.actor.optimizer.step()

        return self.bc_loss.cpu().detach().numpy()

    def saveCheckpoints(self, ajan, model_name):
        torch.save(self.actor.state_dict(), model_name + '\\{}'.format(ajan) + self.actor.name)

    def loadCheckpoints(self, ajan, model_name):
        self.actor.load_state_dict(torch.load(model_name + '\\{}'.format(ajan) + self.actor.name))

    def chooseActionNoNoise(self, state):
        self.actor.eval()
        state = torch.from_numpy(state).float().to(self.actor.device)
        action = self.actor(state)
        return action.cpu().detach().numpy()
