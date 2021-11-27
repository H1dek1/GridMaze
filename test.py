#!/usr/bin/env python

import sys
import grid_maze
import gym

def main():
    """
    Set Environment
    """
    env = gym.make(
            'GridMaze-v0',
            obstacle_positions=[
                [1, 1],
                ],
            collision_penalty=0.0,
            render_dir='./render/'
            )
    
    """
    Set Agent
    """
    maze_map = env.get_zero_map()
    print(maze_map)
    env.draw_heat_map(maze_map, cmap='cool', fname='sample')

    """
    Test
    """
    obs = env.reset()
    env.render()
    done = False
    step_counter = 0

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        step_counter += 1

    print('total steps:', step_counter)


if __name__ == '__main__':
    main()
