import math
import numpy as np
from numpy import random as rnd
from scipy.spatial.distance import cityblock

from abc import ABC, abstractmethod

DOWN, LEFT, UP, RIGHT, STAY = range(5)

class Agent(ABC):

  def __init__(self, name: str):
    self.name = name
    self.observation = None

  def see(self, observation):
    self.observation = observation

  @abstractmethod
  def action(self) -> int:
    raise NotImplementedError()
  

class RandomAgent(Agent):

  def __init__(self, name: str, n_actions: int):
    super(RandomAgent, self).__init__(name)
    self.n_actions = n_actions

  def action(self) -> int:
    return rnd.randint(0, self.n_actions)
  

class GreedyAgent(Agent):

  def __init__(self, name: str, n_actions: int, n_agents: int):
    super(GreedyAgent, self).__init__(name)
    self.n_actions = n_actions
    self.n_agents = n_agents

  def action(self) -> int:
    agents = self.observation[0]
    apples = self.observation[1]
    agent_pos = agents[0]

    closest_apple = self._closest_apple(agent_pos, apples)
    return self._direction_to(agent_pos, closest_apple)

  def _closest_apple(self, pos, apples):
    closest_apple = None
    closest_distance = math.inf
    for apple in apples:
      distance = cityblock(pos, apple)
      if distance < closest_distance:
        closest_distance = distance
        closest_apple = apple
    return closest_apple

  def _direction_to(self, pos, target):
    if target == None:
      return STAY
    x, y = pos
    tx, ty = target
    if x == tx:
      return RIGHT if y < ty else LEFT
    elif y == ty:
      return DOWN if x < tx else UP
    else:
      return RIGHT if y < ty else LEFT