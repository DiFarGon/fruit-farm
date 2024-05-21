import time
from gym import Env
import numpy as np

from environment import Environment
from wrappers import SingleAgentWrapper

from agents import Agent, GreedyAgent, RandomAgent

def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:
  results = np.zeros(n_episodes)

  for episode in range(n_episodes):
    steps = 0
    terminal = False
    observation = environment.reset()
    environment.render()
    time.sleep(0.5)
    while not terminal:
      steps += 1
      agent.see(observation)
      action = agent.action()
      observation, terminal = environment.step(action)
      environment.render()
      time.sleep(0.5)
    environment.close()

    results[episode] = steps

  return results

if __name__ == '__main__':
  environment = Environment(n_agents=1, disaster_prob=0.001)
  environment = SingleAgentWrapper(environment, agent_id=0)

  
  agent = GreedyAgent(name='greedy_agent', n_actions=5, n_agents=1)

  results = {
    agent.name: run_single_agent(environment, agent, n_episodes=100)
  }

  print(results)