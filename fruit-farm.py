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
    observations = environment.reset()
    while not terminal:
      agent.see(observations[0])
      action = agent.action()
      observations, terminal = environment.step(action)
    environment.close()

    results[episode] = environment.total_episode_reward[0]

  return results

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:
  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    terminals = [False for _ in agents]
    observations = environment.reset()
    while not all(terminals):
      for i in range(len(agents)):
        agent = agents[i]
        agent.see(observations[i])
      actions = [agent.action() for agent in agents]
      observations, terminals = environment.step(actions)
    results[episode] = np.sum(environment.total_episode_reward)
    environment.close()

    return results

if __name__ == '__main__':
  environment = Environment(n_agents=4, disaster_prob=0.001)
 
  teams = {
    'random': [RandomAgent(f'random_{i}', environment.action_space[i].n) for i in range(4)],
    'greedy': [GreedyAgent(f'greedy_{i}', environment.action_space[i].n, 4) for i in range(4)]
  }

  results = {}
  for team, agent in teams.items():
    result = run_multi_agent(environment, agent, 100)
    results[team] = result.mean()

  print(results)