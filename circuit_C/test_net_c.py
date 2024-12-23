#Code to test the trained network. 

import torch
import os
import torch.nn as nn
from circuit_env_c import CustomFunctionEnv
import csv

class Actor(nn.Module):  
    def __init__(self, n_input, n_output, n_hidden):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(n_hidden, n_output)
        self.activation3 = nn.Tanh()  # -1 ~ +1
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init_)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)/10
        return x

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


env = CustomFunctionEnv()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netA = Actor(n_input=4, n_hidden=32, n_output=4).to(device)


def load_checkpoint(filename, netA):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        netA.load_state_dict(checkpoint['netA_state_dict'])
        print(f"Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {filename}")


checkpoint_file = "checkpoint_circuit_c_3.pth.tar"
csv_filename = "circuit_C_4.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FOM", "state", "action", "next_state", "reward"])

load_checkpoint(checkpoint_file, netA)


def inference(netA, env, csv_filename,num_steps=30):
    state = torch.tensor(env._random_point_in_range(), dtype=torch.float32).to(device)
    fom =env.FOM(env._denormalize(state[0].item()), env._denormalize(state[1].item()),env._denormalize(state[2].item()),env._denormalize(state[3].item()))
    while fom == 0:
        state = torch.tensor(env._random_point_in_range(), dtype=torch.float32).to(device)
        fom =env.FOM(env._denormalize(state[0].item()), env._denormalize(state[1].item()),env._denormalize(state[2].item()),env._denormalize(state[3].item()))
    for _ in range(num_steps):
        with torch.no_grad():  
            action = netA(state.unsqueeze(0))[0]
            next_state, reward, _, _, info = env.step(state.cpu().numpy(), action.cpu().numpy())
            new_x1,new_x2,new_x3,new_x4 = env._denormalize(next_state[0].item()), env._denormalize(next_state[1].item()),env._denormalize(next_state[2].item()),env._denormalize(next_state[3].item())
            x1,x2,x3,x4 = env._denormalize(state[0].item()), env._denormalize(state[1].item()),env._denormalize(state[2].item()),env._denormalize(state[3].item())
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([info,[x1,x2,x3,x4],action.cpu().numpy(),[new_x1,new_x2,new_x3,new_x4],reward])
            state = torch.tensor(next_state, dtype=torch.float32).to(device)


inference(netA, env, csv_filename, num_steps=400)


