from circuit_env_c import CustomFunctionEnv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os
import sys
import signal
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

import pandas as pd
from datetime import datetime

log_path = 'userlog/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_state_fom.csv'
if not os.path.exists('userlog'):
    os.makedirs('userlog')
#You can see the state and fom added to the replay memory.
with open(log_path, 'w') as f:
    f.write("iteration,x1,x2,x3,x4,fom\n")

class ReplayMemory: 
    def __init__(self,  device='cuda'):
        self.buffer = []
        self.device = device

    def push(self, state, FOM):
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        FOM = torch.tensor(FOM, dtype=torch.float32).to(self.device)
        
        single_data_pair = state, FOM 
        self.buffer.append(single_data_pair)

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        state, FOM = map(torch.stack, zip(*batch))  
        return state,  FOM 
    
    def show(self):
        for idx in range(len(self.buffer)):
            print(f"state: {self.buffer[idx][0]}, FOM: {self.buffer[idx][1]}\n")

    def __len__(self):
        return len(self.buffer)
    

    def get_states_with_min_FOM(self, k=5):
        if len(self.buffer) < k:
            k = len(self.buffer)  # Adjust k if buffer has fewer samples

        # Sort the buffer by FOM value and get the top k states
        sorted_buffer = sorted(self.buffer, key=lambda x: x[1].item(), reverse=True)
        top_k_states = [pair[0] for pair in sorted_buffer[:k]]

        return torch.stack(top_k_states).to(self.device)


def weights_init_(m):  
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):  
    def __init__(self, n_input, n_output, n_hidden):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.activation3 = nn.ReLU()
        self.layer4 = nn.Linear(n_hidden,n_output)
        self.activation4 = nn.Tanh()  # -1 ~ +1
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init_)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x) 
        x = self.layer4(x)
        x = self.activation4(x) 
        return x


class Critic(nn.Module):  
    def __init__(self, n_input, n_output, n_hidden):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.activation3 = nn.ReLU()
        self.layer4= nn.Linear(n_hidden,n_output)
        self.activation4 = nn.Tanh()  
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        return x


# set environment
env = CustomFunctionEnv()  

# set memory
memory = ReplayMemory(device=device) 

# set hyperparameters
batch_size = 64
lr_C = 0.0003
lr_A = 0.0003

gamma = 0.9

epsilon_start = 0.9
epsilon_end = 0.3
epsilon_decay = 0.999  
epsilon = epsilon_start
# set iteration count
warm_up_step = 100 #
start_update_step = warm_up_step
print_freq = 100
max_iter = 1000
save_freq = 100
checkpoint_file = "checkpoint_circuit_c_dnn_4.pth.tar"
# define criterion
criterion = nn.HuberLoss()

# networks
netA = Actor(n_input=4, n_hidden=32, n_output=4).to(device)
netC = Critic(n_input=8, n_hidden=32, n_output=1).to(device)

optimizer_A = torch.optim.Adam(netA.parameters(), lr=lr_A)
optimizer_C = torch.optim.Adam(netC.parameters(), lr=lr_C)

# define target network
netA_target = Actor(n_input=4, n_hidden=32, n_output=4).to(device)
netC_target = Critic(n_input=8, n_hidden=32, n_output=1).to(device)

#Target network initialization - copy parameters of main network
netA_target.load_state_dict(netA.state_dict())
netC_target.load_state_dict(netC.state_dict())


# Added function for soft update
def soft_update(target, source, tau=0.005):
    """
    Smoothly updates the target network towards the source network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#Functions for saving and loading model.
def save_checkpoint(epoch, netA, netC, netA_target, netC_target, optimizer_A, optimizer_C, filename):
    state = {
        'epoch': epoch,
        'netA_state_dict': netA.state_dict(),
        'netC_state_dict': netC.state_dict(),
        'netA_target_state_dict': netA_target.state_dict(),
        'netC_target_state_dict': netC_target.state_dict(),
        'optimizer_A_state_dict': optimizer_A.state_dict(),
        'optimizer_C_state_dict': optimizer_C.state_dict()
    }
    torch.save(state, filename)

def load_checkpoint(filename, netA, netC, netA_target, netC_target, optimizer_A, optimizer_C):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename,  weights_only=True)
        

        netA.load_state_dict(checkpoint['netA_state_dict'])
        netC.load_state_dict(checkpoint['netC_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        

        if 'netA_target_state_dict' in checkpoint:
            netA_target.load_state_dict(checkpoint['netA_target_state_dict'])
            netC_target.load_state_dict(checkpoint['netC_target_state_dict'])
        else:
            print("No target network found in checkpoint, copying from current networks")
            netA_target.load_state_dict(netA.state_dict())
            netC_target.load_state_dict(netC.state_dict())
        
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {filename}")
        return 0


def test(netA): 
   """
   Functions for testing while training
   """
   state = torch.tensor(env._random_point_in_range(), dtype=torch.float32).to(device)
   fom = env.FOM(env._denormalize(state[0].item()), env._denormalize(state[1].item()), 
                 env._denormalize(state[2].item()), env._denormalize(state[3].item()))
   
   while fom == 0: #If fom is 0, the non-0 state is taken as the initial state. This is for prevention.
       state = torch.tensor(env._random_point_in_range(), dtype=torch.float32).to(device)
       fom = env.FOM(env._denormalize(state[0].item()), env._denormalize(state[1].item()),
                    env._denormalize(state[2].item()), env._denormalize(state[3].item()))
       
   for _ in range(20):
       action = netA(state.unsqueeze(0))[0].detach()
       next_state = state + action
       next_state = torch.clamp(next_state,min=0.0,max=1.0)
       fom, _, _, _ = env.step(state.cpu().numpy()) #next_state, reward,  #, action.cpu().numpy()
       new_x1, new_x2, new_x3, new_x4 = [env._denormalize(x.item()) for x in next_state]
       x1, x2, x3, x4 = [env._denormalize(x.item()) for x in state]
       print("fom: ", fom,
             "state: ", [x1,x2,x3,x4])
       state = torch.tensor(next_state, dtype=torch.float32).to(device)

def cal_reward(old_FOM, new_FOM):
    """
    Calculate reward based on the change in FOM values
    Args:
        old_FOM: Previous FOM values (tensor)
        new_FOM: New FOM values (tensor)
    Returns:
        rewards: Tensor of calculated rewards
    """
    # Calculate difference
    diff = new_FOM - old_FOM

    rewards = torch.where(diff >= 0, 2 * diff, -0.05)
    return rewards






start_iter = load_checkpoint(checkpoint_file, netA, netC, netA_target, netC_target, optimizer_A, optimizer_C)
for n_iter in range(start_iter, max_iter):

    if n_iter <= warm_up_step: #Preparatory process for collecting data   
    #Save (state, fom) to replay memory
        state = torch.tensor([random.uniform(0,1) for _ in range(4)], dtype=torch.float32).to(device)
        fom, _, _, _ = env.step(state.cpu().numpy()) 
        x1, x2, x3, x4 = env._denormalize(state[0].item()), env._denormalize(state[1].item()), env._denormalize(state[2].item()), env._denormalize(state[3].item())
        #You can see the state and fom added to the replay memory.
        with open(log_path, 'a') as f:
            f.write(f"{n_iter},{x1},{x2},{x3},{x4},{fom}\n")
            f.flush()

        memory.push(state.cpu().numpy(), fom)

    if n_iter > start_update_step:

        #3
        #weight initialization of actor network
        if n_iter <= 200:
            netA.initialize_weights()

       #4
       #Pseudo sampling: (state, action, next_state, reward) = (state_batch1, state_batch2-state_batch1, state_batch2, reward_batch) 
        state_batch1 , FOM_batch1 = memory.sample(batch_size)
        state_batch2 , FOM_batch2 = memory.sample(batch_size)
        state1_denorm = np.array([[env._denormalize(x.item()) for x in state] for state in state_batch1])
        state2_denorm = np.array([[env._denormalize(x.item()) for x in state] for state in state_batch2])
        action_denorm = state2_denorm - state1_denorm
        action_batch = torch.tensor(np.array([[env._normalize(x) for x in action] for action in action_denorm]), 
                                dtype=torch.float32).to(device)
        state_batch = state_batch1
        next_state_batch = state_batch2
        reward_batch = cal_reward(FOM_batch1, FOM_batch2)
        reward_batch = reward_batch.unsqueeze(1)

        mean_reward = reward_batch.mean()
        std_reward = reward_batch.std()
        #Add 1e-5 to prevent std_reward being 0.
        reward_batch_normalized = (reward_batch - mean_reward) / (std_reward + 1e-5) 

        #5,6
        #Network training
        #This RL uses target networks.
        #critic, actor network training.
        next_pi = netA_target(next_state_batch) 
        next_q_value = netC_target(next_state_batch, next_pi)  

        q_value = netC(state_batch, action_batch)
        loss_C = torch.mean(criterion(q_value, reward_batch_normalized + gamma * next_q_value))

        optimizer_C.zero_grad()
        loss_C.backward()
        optimizer_C.step()

  
        pi = netA(state_batch)
        loss_A = -torch.mean(netC(state_batch, pi)) + torch.mean(torch.maximum(torch.abs(state_batch + pi), torch.tensor(1))) - 1
        
        optimizer_A.zero_grad()
        loss_A.backward()
        optimizer_A.step()

        #target network update
        soft_update(netA_target, netA)
        soft_update(netC_target, netC)

        #just for monitoring
        if n_iter % 100 == 0:  
            print("\n" + "="*50)
            print(f"Iteration {n_iter} Monitoring:")
            print("\nPseudo Sampling 모니터링:")
            for i in range(min(5, batch_size)): 
                print(f"\nSample {i}:")
                state1_denorm = [env._denormalize(x.item()) for x in state_batch1[i]]
                state2_denorm = [env._denormalize(x.item()) for x in state_batch2[i]]
                action_denorm = [env._denormalize(x) for x in action_batch[i].cpu().numpy()]
                
                print(f"State 1: {[f'{x}' for x in state1_denorm]}")
                print(f"State 2: {[f'{x}' for x in state2_denorm]}")
                print(f"Action (diff denorm): {[f'{x}' for x in action_denorm]}")
                print(f"FOM 1: {FOM_batch1[i].item():.4f}")
                print(f"FOM 2: {FOM_batch2[i].item():.4f}")
                print(f"Reward: {reward_batch[i].item():.4f}")

            print("\n네트워크 Loss 모니터링:")
            print(f"Critic Loss: {loss_C.item():.4f}")
            print(f"Actor Loss: {loss_A.item():.4f}")
            
            print("\nQ-value 통계:")
            print(f"Q-value mean: {q_value.mean().item():.4f}")
            print(f"Q-value std: {q_value.std().item():.4f}")
            print(f"Next Q-value mean: {next_q_value.mean().item():.4f}")
            print("="*50 + "\n")
        if n_iter % 100 == 0:  
            print(f"\nIteration {n_iter} Monitoring:")
            for i in range(min(5, batch_size)):
                print(f"Sample {i}:")
                print(f"Predicted Q-value: {q_value[i].item():.4f}")
                print(f"Actual reward: {reward_batch[i].item():.4f}")
                print(f"Next state Q-value: {next_q_value[i].item():.4f}")
                state1_denorm = [env._denormalize(x.item()) for x in state_batch[i]]
                state2_denorm = [env._denormalize(x.item()) for x in next_state_batch[i]]
                print(f"FOM1: {env.FOM(state1_denorm[0], state1_denorm[1], state1_denorm[2], state1_denorm[3]):.4f}")
                print(f"FOM2: {env.FOM(state2_denorm[0], state2_denorm[1], state2_denorm[2], state2_denorm[3]):.4f}\n")


        if (n_iter + 1) % print_freq == 0:
            print(f"Current epsilon: {epsilon:.4f}")
            print('[{}/{}] Loss C: {:.3f}    Loss A: {:.3f}'.format(n_iter + 1, max_iter, float(loss_C), float(loss_A)))
            test(netA)
            print()



        # 7~14
        # Create elite solution 
        # elite solution = (State with the highest FOM among existing states) + netA(State with the highest FOM among existing states)     
        k = 1
        if n_iter <= 200:
            state_es_batch = memory.get_states_with_min_FOM(k)

            action_es_batch = netA(state_es_batch)

            state_sample_batch = state_es_batch + action_es_batch

            # Process for saving to replay memory
            for state in state_sample_batch:
                state = state.unsqueeze(0)
                next_state = state.squeeze(0)

                next_state = torch.clamp(next_state, min = 0.0, max = 1.0)

                fom, _, _, _ = env.step(next_state.detach().cpu().numpy())
                x1, x2, x3, x4 = env._denormalize(next_state[0].item()), env._denormalize(next_state[1].item()), env._denormalize(next_state[2].item()), env._denormalize(next_state[3].item())

                # You can see the state and fom added to the replay memory.
                with open(log_path, 'a') as f:
                    f.write(f"{n_iter},{x1},{x2},{x3},{x4},{fom}\n")
                    f.flush()  
                    
                memory.push(next_state.detach().cpu().numpy(), fom)       
    #save model at checkpoint
    if (n_iter + 1) % save_freq == 0:
        save_checkpoint(n_iter + 1, netA, netC, netA_target, netC_target, optimizer_A, optimizer_C, checkpoint_file)


#There is no need for separate test code for inference. Only through the training process can we achieve the best fom and state.

