#!/usr/bin/env python
import numpy as np
import gym
import grid_maze

env = gym.make(
        'GridMaze-v0',
        width=4,
        height=4,
        start_position=np.array([0, 0]),
        goal_position=np.array([3, 3]),
        obstacle_positions=[
            [1, 1],
            [2, 2],
            ],
        reward_map={
            (2, 2): 1,
            }
        )

env.debug()
