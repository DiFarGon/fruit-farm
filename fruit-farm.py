import time
from typing import Sequence
from gym import Env
import numpy as np

from environment import Environment
from wrappers import SingleAgentWrapper

from agents import Agent, CooperativeAgent, GreedyAgent, RandomAgent

def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:
  results = np.zeros((n_episodes, 2))

  for episode in range(n_episodes):
    steps = 0
    terminal = False
    observations = environment.reset()
    while not terminal:
      agent.see(observations[0])
      action = agent.action()
      observations, terminals = environment.step(action)
      terminal = terminals[0]
      steps += 1
    environment.close()

    results[episode] = [environment.total_episode_reward[0], steps]

  return results

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:
  results = np.zeros((n_episodes, 2))

  for episode in range(n_episodes):
    steps = 0
    terminals = [False for _ in agents]
    observations = environment.reset()
    # environment.render()
    # time.sleep(1)
    while not all(terminals):
      for i in range(len(agents)):
        agent = agents[i]
        agent.see(observations[i])
      actions = [agent.action() for agent in agents]
      observations, terminals = environment.step(actions)
      steps += 1
      # environment.render()
      # time.sleep(1)
    results[episode] = [environment.total_episode_reward[0], steps]
    environment.close()

  return results

if __name__ == '__main__':
  environment = Environment(n_agents=4, n_apples=10, disaster_prob=0.001)
 
  teams = {
    'random': [RandomAgent(f'random_{i}', environment.action_space[i].n) for i in range(4)],
    #'greedy': [GreedyAgent(f'greedy_{i}', environment.action_space[i].n, 4) for i in range(4)],
    #'cooperative': [CooperativeAgent(f'cooperative_{i}', environment.action_space[i].n, 4, 5) for i in range(4)],
  }

  results = {}
  for team, agent in teams.items():
    result = run_multi_agent(environment, agent, 100)
    print(result.shape)
    results[team] = {
      'reward': result[:, 0].mean(),
      'steps': result[:, 1].mean()
    }

  print(results)


# sacrificial lamb: if apples fall below a threshold, the agent the furthest away
# from apples will sacrifice itself to feed the others