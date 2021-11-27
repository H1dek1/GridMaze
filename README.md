# GridMaze
Grid World Maze problem with the format of OpenAI gym

![0042](https://user-images.githubusercontent.com/56115620/143460416-af1f33d8-0b06-41cd-80e1-c6db56e16910.png)

# Usage

```python
import gym
import grid_maze

# make instance
env = gym.make(
        'GridMaze-v0',
        render_dir='./render/'
        )

# you can get map
maze_map = env.get_zero_map()
print(maze_map)
# save maze_map as 'heat_map.png'
env.draw_heat_map(maze_map, cmap='gray', fname='heat_map')

# start simulation
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
```

