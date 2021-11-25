from gym.envs.registration import register

register(
        id='GridMaze-v0',
        entry_point='grid_maze.environment:GridMaze'
        )
