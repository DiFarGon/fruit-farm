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
  

class SeeingAgent(Agent):

  def __init__(self, name: str, n_actions: int, n_agents: int):
    super(SeeingAgent, self).__init__(name)
    self.n_actions = n_actions
    self.n_agents = n_agents

  def _distance_to_apples(self, pos, apples):
    distances = [cityblock(pos, apple) for apple in apples]
    return distances

  def _closest_apple(self, pos, apples):
    if pos == None:
      return None, math.inf
    closest_apple = None
    closest_distance = math.inf
    for apple in apples:
      distance = cityblock(pos, apple)
      if distance < closest_distance:
        closest_distance = distance
        closest_apple = apple
    return closest_apple, closest_distance

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

  
class GreedyAgent(SeeingAgent):

  def __init__(self, name: str, n_actions: int, n_agents: int):
    super(GreedyAgent, self).__init__(name, n_actions, n_agents)

  def action(self) -> int:
    agents = self.observation[1]
    apples = self.observation[2]
    agent_pos = agents[0]

    closest_apple, _ = self._closest_apple(agent_pos, apples)
    return self._direction_to(agent_pos, closest_apple)
    

class CooperativeAgent(SeeingAgent):

  def __init__(self, name: str, n_actions: int, n_agents: int, max_hunger: int):
    super(CooperativeAgent, self).__init__(name, n_actions, n_agents)
    self.max_hunger = max_hunger

  def action(self) -> int:
    hunger = self.observation[0]
    agents = self.observation[1]
    apples = self.observation[2]
    agent_pos = agents[0]

    closest_apple, distance = self._closest_apple(agent_pos, apples)
    while not self._agent_is_closest(closest_apple, agents, apples, distance):
      apples.remove(closest_apple)
      closest_apple, distance = self._closest_apple(agent_pos, apples)
    
    if distance == 1 and self.max_hunger > hunger + 1:
      return STAY
    
    return self._direction_to(agent_pos, closest_apple)
  
  def _agent_is_closest(self, apple_pos, agents, apples, distance):
    for agent in agents[1:]:
      other_closest_apple, other_distance = self._closest_apple(agent, apples)
      if other_closest_apple == apple_pos and distance > other_distance:
        return False
    return True