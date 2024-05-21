import time
from typing import Sequence
from gym import Env
import numpy as np

from environment import Environment
from wrappers import SingleAgentWrapper

from agents import Agent, GreedyAgent, RandomAgent

def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:
  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    terminal = False
    observation = environment.reset()
    while not terminal:
      agent.see(observation)
      action = agent.action()
      observation, terminal = environment.step(action)
    environment.close()

    results[episode] = environment.total_episode_reward[0]

  return results

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:
  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    steps = 0
    terminals = [False for _ in agents]
    observations = environment.reset()
    while not all(terminals):
      pass
    results[episode] = steps

if __name__ == '__main__':
  environment = Environment(n_agents=1, disaster_prob=0.001)
  environment = SingleAgentWrapper(environment, agent_id=0)

  
  agent1 = GreedyAgent(name='greedy_agent', n_actions=5, n_agents=1)
  agent2 = RandomAgent(name='random_agent', n_actions=5)

  results = {
    agent1.name: run_single_agent(environment, agent1, n_episodes=100).mean(),
    agent2.name: run_single_agent(environment, agent2, n_episodes=100).mean()
  }

  print(results)