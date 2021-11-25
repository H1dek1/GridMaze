import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridMaze(gym.Env):
    def __init__(self, width=3, height=3, start_position=np.array([0, 0]), goal_position=np.array([2, 2]), obstacle_positions=[], reward_map={}):
        """
        set grid world
        """
        self.xmin = 0
        self.xmax = width - 1
        self.ymin = 0
        self.ymax = height - 1
        
        self.map = np.zeros((width, height))
        self.map[start_position[0], start_position[1]] = 1
        self.map[goal_position[0], goal_position[1]] = 2
        for pos in obstacle_positions:
            self.map[pos[0], pos[1]] = -1

        self.reward_map = np.zeros((width, height))
        for pos, reward in reward_map.items():
            self.reward_map[pos[0], pos[1]] = reward

        # self.observation_pace = gym.spaces.Discrete(height*width)

        """
        set action space
        """
        self.action_list = {}
        self.action_list[0] = np.array([ 1, 0])
        self.action_list[1] = np.array([ 0, 1])
        self.action_list[2] = np.array([-1, 0])
        self.action_list[3] = np.array([ 0,-1])

        self.action_space = gym.spaces.Discrete(len(self.action_list))

        self.start_pos = start_position
        self.goal_pos = goal_position

    def reset(self):
        self.pos = self.start_pos.copy()
        self.step_counter = 0
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.imshow(self.map, cmap='gray', vmin=-1, vmax=0, origin='lower')

        start_circle = patches.Circle(xy=self.start_pos, radius=0.4, ec='k', fill=False)
        goal_circle = patches.Circle(xy=self.goal_pos, radius=0.4, ec='k', fill=False)
        self.ax.text(self.start_pos[0], self.start_pos[1], s='start', ha='center', va='center')
        self.ax.text(self.goal_pos[0], self.goal_pos[1], s='goal', ha='center', va='center')
        self.ax.add_patch(start_circle)
        self.ax.add_patch(goal_circle)
        self.agent_circle = None
        return self.pos

    def step(self, action_id):
        tmp_pos = self.pos + self.action_list[action_id]
        if self.xmin <= tmp_pos[0] <= self.xmax \
                and self.ymin <= tmp_pos[1] <= self.ymax \
                and self.map[tmp_pos[0], tmp_pos[1]] != -1:
            self.ax.plot(
                    [self.pos[0], tmp_pos[0]],
                    [self.pos[1], tmp_pos[1]],
                    c='r'
                    )
            self.pos = tmp_pos.copy()
            reward = self.reward_map[self.pos[0], self.pos[1]]
            if self.map[self.pos[0], self.pos[1]] == 2:
                done = True
            else:
                done = False

        else:
            reward = -1
            done = False

        info = {}
        self.step_counter += 1
        return self.pos, reward, done, info

    def render(self, mode='rgb_array'):
        if self.agent_circle is not None:
            self.agent_circle.remove()
        self.agent_circle = patches.Circle(xy=self.pos, radius=0.1, fc='r')
        self.ax.add_patch(self.agent_circle)
        self.fig.savefig(f'render/{self.step_counter:04}.png')

    def debug(self):
        print('x range:', self.xmin, self.xmax)
        print('y range:', self.ymin, self.ymax)
        print('map')
        print(self.map)
        print('reward map')
        print(self.reward_map)
        print('action list')
        print(self.action_list)

        print('reset')
        obs = self.reset()
        print('obs:', obs)

        done = False
        while not done:
            action = self.action_space.sample()
            obs, reward, done, _ = self.step(action)
            self.render()
            print('action:', action, ', pos:', obs)

        print('total steps:', self.step_counter)
