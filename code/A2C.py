import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env = gym.make('CartPole-v1').unwrapped

state_number = env.observation_space.shape[0]
action_number = env.action_space.n
# print(action_number) 2

LR_A = 0.005    # learning rate for actor
LR_C = 0.01     # learning rate for critic
Gamma = 0.9

Switch=0        # 训练、测试切换标志

