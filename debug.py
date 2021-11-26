#!/usr/bin/env python
import numpy as np
import gym
import grid_maze

width = 4
height = 1
env = gym.make(
        'GridMaze-v0',
        flatten_state=True,
        width=width,
        height=height,
        start_position=np.array([0, 0]),
        goal_position=np.array([2, 0]),
        obstacle_positions=[
            # [1, 1],
            # [3, 1],
            # [1, 3],
            # [1, 3],
            # [2, 4],
            ],
        reward_map={
            # (2, 2): 10,
            },
        render_dir='./render/'
        )

env.debug()
env.get_zero_map()
state_value = np.random.rand(width, height)
env.plot_state_value_function(state_value, cmap='YlOrRd', fname='state_value')
