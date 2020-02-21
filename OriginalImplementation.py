# Environment
!pip install git+https://github.com/eleurent/highway-env#egg=highway-env  > /dev/null 2>&1
import gym
import highway_env

# Agent
!pip install git+https://github.com/eleurent/rl-agents#egg=rl-agents > /dev/null 2>&1
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm import tnrange
!git clone https://github.com/eleurent/highway-env.git > /dev/null 2>&1
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
sys.path.insert(0, './highway-env/scripts/')
from utils import record_videos, show_videos, capture_intermediate_frames

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
from pathlib import Path
import base64
import pyvirtualdisplay

def record_videos(env, path="videos"):
    return Monitor(env, path, force=True, video_callable=lambda episode: True)


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def capture_intermediate_frames(env):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame


# Environment
# !pip3 install git+https://github.com/eleurent/highway-env#egg=highway-env  > /dev/null 2>&1
import numpy as np
import gym
import highway_env
import pytest
import time
# from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm import tnrange
from rl_agents.agents.deep_q_network.pytorch import DQNAgent

from gym.wrappers import Monitor

# Visualization
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tnrange
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor
import base64

# IO
from pathlib import Path

torch = pytest.importorskip("torch")

display = Display(visible=0, size=(1400, 900))
display.start()

def show_video():
    html = []
    for mp4 in Path("video").glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

def test_env(env_name, model_type, n_episodes, layer_dim, gamma):

    env = gym.make(env_name)
    env = Monitor(env, './video', force=True, video_callable=lambda episode: True)
    
    if model_type == "DuelingNetwork":
        agent_config = {
            "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
            "model": {
                "type": "DuelingNetwork",
                "layers": [layer_dim, layer_dim]
            },
            "gamma": gamma,
            "n_steps": 1,
            "batch_size": 32,
            "memory_capacity": 15000,
            "target_update": 50,
            "exploration": {
                "method": "EpsilonGreedy",
                "tau": 6000,
                "temperature": 1.0,
                "final_temperature": 0.05
            }
        }
        
    elif model_type == "MultiLayerPerceptron":
        agent_config = {
            "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
            "model": {
                "type": "MultiLayerPerceptron",
                "layers": [layer_dim, layer_dim]
            },
            "gamma": gamma,
            "n_steps": 1,
            "batch_size": 64,
            "memory_capacity": 15000,
            "target_update": 256,
            "exploration": {
                "method": "EpsilonGreedy",
                "tau": 6000,
                "temperature": 1.0,
                "final_temperature": 0.05
            }
        }
       
    agent = DQNAgent(env, config=agent_config)
    
    print("config", agent.config)

    done = False
    
    #print('n', n)
    
    rewards = []
    episode_rewards = [] 
    """for episode in tnrange(3, desc="Test episodes"):
        obs, done = env.reset(), False
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
    env.close()"""
    
    elapsed_time = []
    
    for episode in tnrange(n_episodes, desc="Test episodes"):
        start_time = time.time()
        agent.seed(episode)
        agent.reset()
        state, done = env.reset(), False
        #env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        print("Begining Episode", episode)
        while not done: 
            #env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
            #print("state", episode, state)
            actions = agent.plan(state)

            action = actions[0]
            #print("action", episode, action)
            assert action is not None

            next_state, reward, done, info = env.step(action)

            #print("next_state", episode, next_state)
            #print("reward", episode, reward)
            #print("done", episode, done)

            episode_rewards.append(reward)
            agent.record(state, action, reward, next_state, done, info)
            state = next_state
        rewards.append(episode_rewards)
        episode_rewards = []
        state = env.reset()
        done = False
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        
#     print(agent.eval())   
    #show_video()
    '''assert (len(agent.memory) == n or
            len(agent.memory) == agent.config['memory_capacity'])'''
    env.close()
    
    return rewards, elapsed_time

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_returns(total_returns, labels, title):
    for j, r in enumerate(total_returns):
        plt.plot([i for i in range(0,len(r))], r, label=labels[j])
    plt.xlabel('Number Episodes')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()
    
def plot_time(total_returns, labels, title):
#     total_returns = np.cumsum(total_returns)
    for j, r in enumerate(total_returns):
        plt.plot([i for i in range(0,len(r))], r, label=labels[j])
    plt.xlabel('Number Episodes')
    plt.ylabel('Learning Time')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()


n = 1000
dn_rewards_mlp, dn_time_mlp = test_env('intersection-v0', "MultiLayerPerceptron", n, 128, 0.7)
dn_rewards_dn, dn_time_dn = test_env('intersection-v0', "DuelingNetwork", n, 128, 0.7)

dn_returns_mlp = [np.sum(r) for r in dn_rewards_mlp]
dn_returns_dn = [np.sum(r) for r in dn_rewards_dn]

dn_learntime_mlp = [np.sum(r) for r in dn_time_mlp]
dn_learntime_dn = [np.sum(r) for r in dn_time_dn]

plot_returns([running_mean(dn_returns_mlp, 10), running_mean(dn_returns_dn, 10)],["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Returns vs Episodes \n Running Mean N=10")
plot_returns([running_mean(dn_returns_mlp, 50), running_mean(dn_returns_dn, 50)], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Returns vs Episodes \n Running Mean N=50")
plot_returns([running_mean(dn_returns_mlp, 100), running_mean(dn_returns_dn, 100)], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Returns vs Episodes \n Running Mean N=100")

plot_returns([dn_returns_mlp, dn_returns_dn], ["SingleNetwork (MLP)", "DuelingNetwork (2 MLPs)"], "Intersection V0 - Returns vs Episodes")
