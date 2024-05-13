import copy
import time
import gym

import numpy as np
import numpy.random as rnd

from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from gym.envs.classic_control import rendering

class Environment(gym.Env):
  

  def __init__(self, grid_shape=(10, 10), n_apples=10, n_agents=0, disaster_prob=0.05, max_steps=100):
    self._grid_shape = grid_shape
    self._n_step = 0
    self.n_apples = n_apples
    self.disaster_prob = disaster_prob
    self._max_steps = max_steps

    self._apple_id_counter = 0

    self.n_agents = n_agents

    self.apples = {}

    self.viewer = rendering.SimpleImageViewer()

    self._create_grid()
  

  def _spawn_apples(self):
    while self._apple_id_counter < self.n_apples:
      x = np.random.randint(0, self._grid_shape[0])
      y = np.random.randint(0, self._grid_shape[1])
      pos = (x, y)
      self._create_apple(pos)

  
  def _is_apple(self, pos):
    (x, y) = pos
    cell = self._grid[x][y]
    if isinstance(cell, str):
      return cell[:5] == 'apple'
    return False
  

  def _is_empty(self, pos):
    (x, y) = pos
    cell = self._grid[x][y]
    return cell == 0
  

  def _delete_apple(self, apple_tag):
    (x, y) = self.apples.pop(apple_tag)
    self._grid[x][y] = 0


  def _create_grid(self):
    self._grid = [[0 for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
    self._spawn_apples()


  def _disaster(self):
    if rnd.choice([True, False], p=[self.disaster_prob, 1-self.disaster_prob]):
      for apple in list(self.apples.keys()):
        if rnd.choice([True, False], p=[DISASTER_INTENSITY, 1-DISASTER_INTENSITY]):
          self._delete_apple(apple)


  def _adjacent_positions(self, pos):
    adjacency = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not i == j == 0]
    (x, y) = pos
    adjacent_pos = []
    for dx, dy in adjacency:
      if 0 <= x + dx < self._grid_shape[0] and 0 <= y + dy < self._grid_shape[1]:
        adjacent_pos.append((x + dx, y + dy))
    return adjacent_pos


  def _n_adjacent_apples(self, pos):
    n_apples = 0
    for adjpos in self._adjacent_positions(pos):
      if self._is_apple(adjpos):
        n_apples += 1
    return n_apples


  def _create_apple(self, pos):
    x, y = pos
    tag = f'apple{self._apple_id_counter}'
    if self._grid[x][y] == 0:
      self._grid[x][y] = tag
      self.apples[tag] = pos
      self._apple_id_counter += 1


  def _grow_apples(self):
    for x in range(self._grid_shape[0]):
      for y in range(self._grid_shape[1]):
        pos = (x, y)
        if self._is_empty(pos):
          growth_prob = ((self._n_adjacent_apples(pos) / 8) / 10) * GROWTH_RATE
          print(growth_prob)
          if rnd.choice([True, False], p=[growth_prob, 1-growth_prob]):
            self._create_apple(pos)
            

  def __draw_base_img(self):
    self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=CELL_FILL_COLOR)


  def step(self):
    if self._n_step >= self._max_steps:
      return True
    
    self._disaster()
    self._grow_apples()

    self._n_step += 1

    return False
  

  def render(self):
    self.__draw_base_img()

    img = copy.copy(self._base_img)

    for apple in self.apples.values():
      draw_circle(img, apple, cell_size=CELL_SIZE, fill=APPLE_COLOR)

    img = np.asarray(img)
    self.viewer.imshow(img)

    time.sleep(0.5)
    
    return self.viewer.isopen

CELL_SIZE = 50
CELL_FILL_COLOR = 'white'

APPLE_COLOR = 'red'

DISASTER_INTENSITY = 0.7

GROWTH_RATE = 1