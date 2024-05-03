import copy
import time
import gym

import numpy as np

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from gym.envs.classic_control import rendering

class Environment(gym.Env):
  

  def __init__(self, grid_shape=(10, 10), n_apples=10):
    self._grid_shape = grid_shape
    self.n_apples = n_apples

    self.apples = {}

    self.viewer = rendering.SimpleImageViewer()

    self._create_grid()
  

  def _spawn_apples(self):
    apples = 0
    while apples < self.n_apples:
      x = np.random.randint(0, self._grid_shape[0])
      y = np.random.randint(0, self._grid_shape[1])
      tag = f'apple{apples}'
      if self._grid[x][y] == 0:
        self._grid[x][y] = tag
        self.apples[tag] = (x, y)
        apples += 1


  def _create_grid(self):
    self._grid = [[0 for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
    self._spawn_apples()


  def __draw_base_img(self):
    self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=CELL_FILL_COLOR)


  def render(self):
    self.__draw_base_img()

    img = copy.copy(self._base_img)

    print(self.apples)
    for apple in self.apples.values():
      draw_circle(img, apple, cell_size=CELL_SIZE, fill=APPLE_COLOR)

    img = np.asarray(img)
    self.viewer.imshow(img)

    time.sleep(10)
    
    return self.viewer.isopen

CELL_SIZE = 50
CELL_FILL_COLOR = 'white'

APPLE_COLOR = 'red'