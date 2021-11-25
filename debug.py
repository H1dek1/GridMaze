#!/usr/bin/env python
import numpy as np
import gym
import grid_maze

width = 4
height = 5
env = gym.make(
        'GridMaze-v0',
        width=width,
        height=height,
        start_position=np.array([0, 1]),
        goal_position=np.array([3, 2]),
        obstacle_positions=[
            [1, 1],
            [3, 1],
            [1, 3],
            [1, 3],
            [2, 4],
            ],
        reward_map={
            (2, 2): 10,
            },
        render_dir='./render/'
        )

env.debug()
state_value = np.random.rand(width, height)
env.plot_state_value_function(state_value, cmap='bwr')
