#This code utilizes a Deep Deterministic Policy Gradient (DDPG) and does not typically use the target network in the DDPG.  
#Main code

from circuit_env_c import CustomFunctionEnv  # Load custom environment
import torch  # Import PyTorch library
import torch.nn as nn  # Import neural network module from PyTorch
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
import random  # Import random module for random sampling
import os  # Import OS module for operating system functionalities

# CUDA setup: Use CUDA if GPU is available
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

# Replay Memory Class: Buffer for storing and sampling experiences in DDPG
class ReplayMemory:  
    def __init__(self, device='cuda'):
        self.buffer = []  # Buffer to store state, action, reward, and next state
        self.device = device

    def push(self, state, action, reward, next_state):
        # Convert data to tensors and store them in the buffer
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        single_data_pair = state, action, reward, next_state
        self.buffer.append(single_data_pair)

    def sample(self, batch_size):
        # Randomly sample a batch from the buffer
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)

# Function to initialize weights in the neural network
def weights_init_(m):  
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)  # Xavier initialization
        nn.init.constant_(m.bias, 0)  # Bias initialization

# Actor Neural Network: Determines actions based on the state
class Actor(nn.Module): 
    def __init__(self, n_input, n_output, n_hidden):
        super(Actor, self).__init__()
        # Define input, hidden, and output layers
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(n_hidden, n_output)
        self.activation3 = nn.Tanh()  # Output range set to -1 ~ +1
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init_)  # Initialize weights

    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x) / 10  # Adjust output range to -0.1 ~ 0.1
        return x

# Critic Neural Network: Evaluates state-action pairs
class Critic(nn.Module):  
    def __init__(self, n_input, n_output, n_hidden):
        super(Critic, self).__init__()
        # Define input, hidden, and output layers
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(n_hidden, n_output)
        self.activation3 = nn.Tanh() 
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init_)  # Initialize weights

    def forward(self, state, action):
        # Concatenate state and action, then pass through the network
        x = torch.cat([state, action], dim=1)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        return x

# Initialize environment
env = CustomFunctionEnv()  

# Initialize replay memory
memory = ReplayMemory(device=device)  

# Set hyperparameters
batch_size = 256
lr_C = 0.0001  # Learning rate for Critic
lr_A = 0.0001  # Learning rate for Actor
epsilon = 0.3  # Exploration probability
gamma = 0.9  # Discount factor

# Training and checkpoint settings
warm_up_step = 2000  # Initial exploration steps
start_update_step = batch_size  # Start updating network after this step
print_freq = 100  # Print frequency
max_iter = 4000  # Maximum iterations
save_freq = 100  # Checkpoint save frequency
checkpoint_file = "checkpoint_circuit_c_3.pth.tar" #You can edit the file name. 

# Define loss function
criterion = nn.L1Loss()

# Initialize neural networks
netA = Actor(n_input=4, n_hidden=32, n_output=4).to(device)
netC = Critic(n_input=8, n_hidden=32, n_output=1).to(device)

# Define optimizers
optimizer_A = torch.optim.Adam(netA.parameters(), lr=lr_A)
optimizer_C = torch.optim.Adam(netC.parameters(), lr=lr_C)

# Save checkpoint function
def save_checkpoint(epoch, netA, netC, optimizer_A, optimizer_C, filename="checkpoint_circuit_c_1.pth.tar"):
    state = {
        'epoch': epoch,
        'netA_state_dict': netA.state_dict(),
        'netC_state_dict': netC.state_dict(),
        'optimizer_A_state_dict': optimizer_A.state_dict(),
        'optimizer_C_state_dict': optimizer_C.state_dict()
    }
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch}")

# Load checkpoint function
def load_checkpoint(filename, netA, netC, optimizer_A, optimizer_C):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        netA.load_state_dict(checkpoint['netA_state_dict'])
        netC.load_state_dict(checkpoint['netC_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {filename}")
        return 0

# Training loop
start_iter = load_checkpoint(checkpoint_file, netA, netC, optimizer_A, optimizer_C)

for n_iter in range(start_iter, max_iter):
    if n_iter % 100 == 0:
        # Reset state periodically
        state = torch.tensor([random.uniform(0, 1) for _ in range(4)], dtype=torch.float32).to(device)

    # Select action: Explore randomly during warm-up steps, exploit policy later
    if n_iter < warm_up_step:
        action = torch.tensor(env.random_action(), dtype=torch.float32).to(device)
    else:
        if random.random() > epsilon:
            action = netA(state.unsqueeze(0))[0].detach()
        else:
            action = torch.tensor(env.random_action(), dtype=torch.float32).to(device)

    # Update networks if memory size is sufficient
    if len(memory) > start_update_step:
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size)

        # Normalize rewards to stabilize training
        reward_batch = reward_batch.unsqueeze(1)
        mean_reward = reward_batch.mean()
        std_reward = reward_batch.std()
        reward_batch_normalized = (reward_batch - mean_reward) / (std_reward + 1e-5)

        # Critic network update
        next_pi = netA(next_state_batch)
        q_value = netC(state_batch, action_batch) * (1 - gamma)
        next_q_value = netC(next_state_batch, next_pi) * (1 - gamma)
        loss_C = torch.mean(criterion(q_value, reward_batch_normalized + gamma * next_q_value))

        optimizer_C.zero_grad()
        loss_C.backward()
        optimizer_C.step()

        # Actor network update
        pi = netA(state_batch)
        loss_A = -torch.mean(netC(state_batch, pi)) + torch.mean(torch.maximum(torch.abs(state_batch + pi), torch.tensor(1))) - 1

        optimizer_A.zero_grad()
        loss_A.backward()
        optimizer_A.step()

        # Print losses and test performance at specified intervals
        if (n_iter + 1) % print_freq == 0:
            print('[{}/{}] Loss C: {:.3f}    Loss A: {:.3f}'.format(n_iter + 1, max_iter, float(loss_C), float(loss_A)))

    # Execute action in environment and store result in memory
    next_state, reward, _, _, _ = env.step(state.cpu().numpy(), action.cpu().numpy())
    memory.push(state.cpu().numpy(), action.cpu().numpy(), reward, next_state)
    state = torch.tensor(next_state, dtype=torch.float32).to(device)

    # Save checkpoint at specified intervals
    if (n_iter + 1) % save_freq == 0:
        save_checkpoint(n_iter + 1, netA, netC, optimizer_A, optimizer_C, checkpoint_file)
