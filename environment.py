import copy
import gym

import numpy as np

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from gym.envs.classic_control import rendering

class Environment(gym.Env):
  
  def __init__(self, grid_shape=(10, 10)):
    self._grid_shape = grid_shape
    self.viewer = rendering.SimpleImageViewer()

  
  def __draw_base_img(self):
    self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=CELL_FILL_COLOR)

  def render(self):
    self.__draw_base_img()

    img = copy.copy(self._base_img)

    img = np.asarray(img)
    self.viewer.imshow(img)

CELL_SIZE = 50
CELL_FILL_COLOR = 'white'