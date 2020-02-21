import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import gym, highway_env
import time
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import gym, highway_env

import random
import numpy as np
from collections import deque
import time

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[start]
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, layer_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        print("input dim", input_dim)
        print("output dim", output_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim, layer_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
        
def multiply_list(myList) : 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result  
    
class Agent:

    def __init__(self, env, use_dueling=True, learning_rate=3e-4, gamma=0.99, buffer_size=10000, layer_dim=128, eps = 0.2):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_dueling = use_dueling
        if self.use_dueling:
            self.model = DuelingDQN((multiply_list(list(env.observation_space.shape)),1), env.action_space.n, 128).to(self.device)
        else:
            print(env.observation_space.shape, env.action_space.n)
            self.model = DQN((multiply_list(list(env.observation_space.shape)),1), env.action_space.n, layer_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < self.eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, layer_dim):
    episode_rewards = []
    episode_runtimes = []

    for episode in range(max_episodes):
        start = time.time()
        state = env.reset()
        state = state.reshape(-1)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            #print('next_state', next_state)
            next_state = next_state.reshape(-1)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                episode_runtimes.append(time.time()-start)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards, episode_runtimes


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_returns(total_returns, labels, title):
    for j, r in enumerate(total_returns):
        plt.plot([i for i in range(0,len(r))], r, label=labels[j])
    plt.xlabel('Number Episodes')
    plt.ylabel('Sum of Rewards')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()
    
def plot_time(total_returns, labels, title):
#     total_returns = np.cumsum(total_returns)
    for j, r in enumerate(total_returns):
        plt.plot([i for i in range(0,len(r))], r, label=labels[j])
    plt.xlabel('Number Episodes')
    plt.ylabel('Learning Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()

def train_env(env_id, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, LAYER_DIM, eps):
    env = gym.make(env_id)
    #print("config1", env.config)
    env.configure({"observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": True,
                    "observe_intentions": False
                }})
    env.reset()
    #print("config2", env.config)
    if agent == "DQNAgent":
        agent = Agent(env, use_dueling=False, gamma = 0.7, layer_dim=LAYER_DIM, eps= eps)
    elif agent == "DuelingAgent":
        agent = Agent(env, gamma = 0.3, layer_dim=LAYER_DIM, eps = eps)
    episode_rewards, episode_runtimes = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, LAYER_DIM)
    return episode_rewards, episode_runtimes
    

env_id = "intersection-v0"#"merge-v15"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32
dn_rewards_mlp, dn_time_mlp = train_env('intersection-v0', "DQNAgent", MAX_EPISODES, MAX_STEPS, BATCH_SIZE, 256, 0.2)
dn_rewards_dn, dn_time_dn = train_env('intersection-v0', "DuelingAgent", MAX_EPISODES, MAX_STEPS, BATCH_SIZE, 64, 0.2)

dn_returns_mlp = [np.sum(r) for r in dn_rewards_mlp]
dn_returns_dn = [np.sum(r) for r in dn_rewards_dn]

dn_learntime_mlp = [np.sum(r) for r in dn_time_mlp]
dn_learntime_dn = [np.sum(r) for r in dn_time_dn]

plot_returns([dn_returns_mlp, dn_returns_dn], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Sum of Rewards vs Episodes")
plot_returns([running_mean(dn_returns_mlp, 10), running_mean(dn_returns_dn, 10)],["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Sum of Rewards vs Episodes \n Running Mean N=10")
plot_returns([running_mean(dn_returns_mlp, 50), running_mean(dn_returns_dn, 50)], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Sum of Rewards vs Episodes \n Running Mean N=50")
plot_returns([running_mean(dn_returns_mlp, 100), running_mean(dn_returns_dn, 100)], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Sum of Rewards vs Episodes \n Running Mean N=100")

plot_time([dn_learntime_mlp, dn_learntime_dn], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Merge V0 - Learning Time vs Episodes")
plot_time([running_mean(dn_learntime_mlp, 100), running_mean(dn_learntime_dn, 100)], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Learning Time vs Episodes \n Running Mean N=100")

print(sum(dn_returns_mlp), sum(dn_returns_dn), sum(dn_learntime_mlp), sum(dn_learntime_dn))
