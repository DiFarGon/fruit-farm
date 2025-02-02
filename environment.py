import copy
import gym

import numpy as np
import numpy.random as rnd

from gym import spaces

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from gym.envs.classic_control import rendering

class Environment(gym.Env):
  

  def __init__(self, grid_shape=(10, 10), n_apples=10, n_agents=0, disaster_prob=0.05, growth_rate=1,
               max_steps=500, step_cost=-0.1, eat_reward=5, starve_steps=5, starve_penalty=-20):
    self._grid_shape = grid_shape
    self._n_step = 0
    self.n_apples = n_apples
    self.disaster_prob = disaster_prob
    self._max_steps = max_steps
    self._step_cost = step_cost
    self._eat_reward = eat_reward
    self._last_ate = None
    self._starve_steps = starve_steps
    self._starve_penalty = starve_penalty
    self._growth_rate = growth_rate

    self._apple_id_counter = 0

    self.total_episode_reward = None

    self.n_agents = n_agents
    self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])

    self.apples = {}
    self.agents = {}
    self._agent_dones = [False for _ in range(self.n_agents)]

    self.viewer = rendering.SimpleImageViewer()

    self._is_shy = False

    self._create_grid()
  

  def reset(self):
    self.total_episode_reward = [0 for _ in range(self.n_agents)]
    self._last_ate = [0 for _ in range(self.n_agents)]
    self._agent_dones = [False for _ in range(self.n_agents)]
    self._n_step = 0
    self.apples = {}
    self.agents = {}
    self._apple_id_counter = 0
    self._create_grid()
    return self._get_features()
  

  def _get_features(self):
    features = []
    half_point = int(self._grid_shape[0]/2)
    if self._is_shy:
      for agent_i, pos in enumerate(self.agents.values()):
        every_agent = list(self.agents.values())
        agents = [pos] + every_agent[:agent_i] + every_agent[agent_i+1:]
        apples = list(self.apples.values())
        if agent_i == 0:
          obs_apples = [apple for apple in apples if apple[0] < half_point and apple[1] < half_point]
        if agent_i == 1:
          obs_apples = [apple for apple in apples if apple[0] < half_point and apple[1] >= half_point]
        if agent_i == 2:
          obs_apples = [apple for apple in apples if apple[0] >= half_point and apple[1] < half_point]
        if agent_i == 3:
          obs_apples = [apple for apple in apples if apple[0] >= half_point and apple[1] >= half_point]
        hunger = self._n_step - self._last_ate[agent_i]
        agent_features = [hunger, agents, obs_apples]
        features.append(agent_features)
      return features
    else:
      for agent_i, pos in enumerate(self.agents.values()):
        every_agent = list(self.agents.values())
        agents = [pos] + every_agent[:agent_i] + every_agent[agent_i+1:]
        apples = list(self.apples.values())
        hunger = self._n_step - self._last_ate[agent_i]
        agent_features = [hunger, agents, apples]
        features.append(agent_features)
      return features


  def _spawn_apples(self):
    while self._apple_id_counter < self.n_apples:
      x = rnd.randint(0, self._grid_shape[0])
      y = rnd.randint(0, self._grid_shape[1])
      pos = (x, y)
      self._create_apple(pos)

  
  def _is_apple(self, pos):
    if not self._is_valid(pos): return False
    (x, y) = pos
    cell = self._grid[x][y]
    if isinstance(cell, str):
      return cell[:5] == 'apple'
    return False


  def _create_apple(self, pos):
    tag = f'apple{self._apple_id_counter}'
    if self._is_empty(pos):
      x, y = pos
      self._grid[x][y] = tag
      self.apples[tag] = pos
      self._apple_id_counter += 1

  
  def _spawn_agents(self):
    agent_id = 0
    if self._is_shy != None and self._is_shy:
      half_point = int(self._grid_shape[0]/2)
      match_pos = {0:(0,0), 1:(0,half_point), 2:(half_point,0), 3:(half_point,half_point)}
      while agent_id < self.n_agents:
        x, y = match_pos[agent_id]
        self._spawn_agent((x, y), agent_id)
        agent_id += 1
    else:
      while agent_id < self.n_agents:
        x = rnd.randint(0, self._grid_shape[0])
        y = rnd.randint(0, self._grid_shape[1])
        if self._is_empty((x, y)):
          self._spawn_agent((x, y), agent_id)
          agent_id += 1

  
  def _spawn_agent(self, pos, id):
    x, y = pos
    tag = f'agent{id}'
    self._grid[x][y] = tag
    self.agents[tag] = pos


  def _is_valid(self, pos):
    x, y = pos
    return 0 <= x < self._grid_shape[0] and 0 <= y < self._grid_shape[1]

  def _is_empty(self, pos):
    if not self._is_valid(pos): return False
    (x, y) = pos
    cell = self._grid[x][y]
    return cell == 0
  

  def _delete_apple(self, apple_tag):
    (x, y) = self.apples.pop(apple_tag)
    self._grid[x][y] = 0


  def _create_grid(self):
    self._grid = [[0 for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
    self._spawn_apples()
    self._spawn_agents()


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


  def _grow_apples(self):
    for x in range(self._grid_shape[0]):
      for y in range(self._grid_shape[1]):
        pos = (x, y)
        if self._is_empty(pos):
          growth_prob = ((self._n_adjacent_apples(pos) / 8) / 10) * self._growth_rate
          if rnd.choice([True, False], p=[growth_prob, 1-growth_prob]):
            self._create_apple(pos)


  def _get_agent_tag(self, agent_i):
    return f'agent{agent_i}'
  

  def _get_agent_index(self, agent_tag):
    return int(agent_tag[5:])
  

  def _update_agent_position(self, agent_i, action):
    agent_tag = self._get_agent_tag(agent_i)
    x, y = self.agents[agent_tag]
    new_pos = None
    if action == DOWN:
      new_pos = (x + 1, y)
    elif action == LEFT:
      new_pos = (x, y - 1)
    elif action == UP:
      new_pos = (x - 1, y)
    elif action == RIGHT:
      new_pos = (x, y + 1)
    elif action == STAY:
      new_pos = (x, y)
    if (x, y) == new_pos:
      return 0
    if self._is_empty(new_pos):
      self.agents[agent_tag] = new_pos
      self._grid[new_pos[0]][new_pos[1]] = agent_tag
      self._grid[x][y] = 0
      return 0
    elif self._is_apple(new_pos):
      apple = self._grid[new_pos[0]][new_pos[1]]
      self._delete_apple(apple)
      self.agents[agent_tag] = new_pos
      self._grid[new_pos[0]][new_pos[1]] = agent_tag
      self._grid[x][y] = 0
      self._last_ate[agent_i] = self._n_step
      return 1
    return 0
  

  def _remove_agent(self, agent_i):
    agent_tag = self._get_agent_tag(agent_i)
    x, y = self.agents[agent_tag]
    self._grid[x][y] = 0
    self.agents[agent_tag] = None


  def __draw_base_img(self):
    self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=CELL_FILL_COLOR)


  def step(self, agents_action):
    self._n_step += 1
    
    rewards = [self._step_cost for _ in range(self.n_agents)]
    
    for agent_i, action in enumerate(agents_action):
      if self._agent_dones[agent_i]:
        continue
      if self._n_step - self._last_ate[agent_i] > self._starve_steps:
        _reward = self._starve_penalty
        self._agent_dones[agent_i] = True
        self._remove_agent(agent_i)
        continue
      apple_eaten = self._update_agent_position(agent_i, action)

      _reward = apple_eaten * self._eat_reward
      rewards[agent_i] += _reward

    for i in range(self.n_agents):
      self.total_episode_reward[i] += rewards[i]

    self._disaster()
    self._grow_apples()

    if self._n_step >= self._max_steps or not self.apples:
      for i in range(self.n_agents):
        self._agent_dones[i] = True
    
    return self._get_features(), self._agent_dones
  

  def render(self, mode='human'):
    self.__draw_base_img()

    img = copy.copy(self._base_img)

    for apple in self.apples.values():
      draw_circle(img, apple, cell_size=CELL_SIZE, fill=APPLE_COLOR)

    for agent_i, agent in enumerate(list(self.agents.values())):
      if agent is None:
        continue
      fill_cell(img, agent, cell_size=CELL_SIZE, fill=AGENT_COLOR)
      hunger = self._n_step - self._last_ate[agent_i]
      write_cell_text(img, text=str(hunger), pos=agent, cell_size=CELL_SIZE, fill='white', margin=0.4)

    img = np.asarray(img)
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)
      return self.viewer.isopen
    
  
  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None


  def _make_shy(self):
    self._is_shy = True

  def _unmake_shy(self):
    self._is_shy = False


CELL_SIZE = 50
CELL_FILL_COLOR = 'white'

APPLE_COLOR = 'red'
AGENT_COLOR = 'blue'

DISASTER_INTENSITY = 0.7

DOWN, LEFT, UP, RIGHT, STAY = range(5)