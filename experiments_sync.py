import torch
import gymnasium as gym
import numpy as np
from torch import nn
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import ale_py

from rollout_buffer import RolloutBuffer
from python_dataset import PythonListDataset
from model import PPOActorCritic
from learner import PPOLearning
import time
from tqdm import tqdm
import torch.multiprocessing as mp
from learner_process import learner_process
from worker_process import worker_process

gym.register_envs(ale_py)
env = gym.make("PongNoFrameskip-v4")

env = gym.wrappers.AtariPreprocessing(
env, 
noop_max=30,
frame_skip=4,
screen_size=84,
grayscale_obs=True,
)
env = gym.wrappers.FrameStackObservation(env, 4)
env.reset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PPOActorCritic(6).to(device)
learner = PPOLearning(model, env, device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

import csv
import time

epochs = 200000
csv_file = 'sync_results.csv'

dummy_dict = dict()
dummy_dict['Worker0'] = 0

start_time = time.time()

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    writer.writerow(['epoch', 'time', 'reward'])
    
    for i in range(epochs):
        rollout = learner.create_rollout(dummy_dict, 0)
        learner.one_epoch_of_learning(rollout=rollout, optimizer=optimizer,
                                      )
        score = learner.testing()

        elapsed_time = time.time() - start_time
        writer.writerow([i, elapsed_time, score])
        
        f.flush()
