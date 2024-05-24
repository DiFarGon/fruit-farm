import os
import time
from typing import Sequence
from gym import Env
import numpy as np
from tqdm import tqdm
import argparse

from environment import Environment
from wrappers import SingleAgentWrapper

from utils import compare_results

from agents import Agent, CooperativeAgent, GreedyAgent, NonRedundantRandomAgent, RandomAgent, ShyAgent

def sanitize_path(path):
    invalid_chars = [":", "=", "(", ")", ","]
    for char in invalid_chars:
        path = path.replace(char, "_")
    return path

def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:
  results = np.zeros((n_episodes, 2))

  for episode in tqdm(range(n_episodes), desc=f'Running {agent.name}'):
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

def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int, visual: bool) -> np.ndarray:
  results = np.zeros((n_episodes, 2))

  for episode in tqdm(range(n_episodes), desc=f'Running {agents[0].name[:-2]} agents'):
    steps = 0
    terminals = [False for _ in agents]
    observations = environment.reset()
    if visual:
      environment.render()
      time.sleep(1)
    while not all(terminals):
      for i in range(len(agents)):
        agent = agents[i]
        agent.see(observations[i])
      actions = [agent.action() for agent in agents]
      observations, terminals = environment.step(actions)
      steps += 1
      if visual:
        environment.render()
        time.sleep(1)
    results[episode] = [environment.total_episode_reward[0], steps]
    environment.close()

  print()
  return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run fruit farm simulation')
  parser.add_argument('--grid_shape', type=int, required=True, help="Grid dimension")
  parser.add_argument('--n_agents', type=int, required=True, help="Number of agents")
  parser.add_argument('--n_apples', type=int, required=True, help="Number of apples")
  parser.add_argument('--disaster_probability', type=float, required=True, help="Disaster probability")
  parser.add_argument('--growth_rate', type=float, required=True, help="Growth rate")

  args = parser.parse_args()

  grid_shape = (args.grid_shape, args.grid_shape)
  n_agents = args.n_agents
  n_apples = args.n_apples
  disaster_probability = args.disaster_probability
  growth_rate = args.growth_rate

  environment = Environment(grid_shape=grid_shape, n_agents=n_agents, n_apples=n_apples,
                            disaster_prob=disaster_probability, growth_rate=growth_rate)
 
  teams = {
    # 'random': [RandomAgent(f'random_{i}', environment.action_space[i].n) for i in range(n_agents)],
    #'greedy': [GreedyAgent(f'greedy_{i}', environment.action_space[i].n, n_agents) for i in range(n_agents)],
    # 'non_redudant_random': [NonRedundantRandomAgent(f'non_redundant_random_{i}', environment.action_space[i].n, grid_shape) for i in range(n_agents)],
    'shy': [ShyAgent(f'shy_{i}', environment.action_space[i].n, n_agents) for i in range(n_agents)],
    #'cooperative': [CooperativeAgent(f'cooperative_{i}', environment.action_space[i].n, 4, 5) for i in range(4)],
  }

  steps = {}
  scores = {}
  for team, agent in teams.items():
    environment._make_shy() if team == "shy" else environment._unmake_shy()
    result = run_multi_agent(environment, agent, 100, visual=False)
    scores[team] = result[:, 0]
    steps[team] = result[:, 1]

  path = f'results/grid_shape={grid_shape}:n_agents={n_agents}:n_apples={n_apples}:disaster_probability={disaster_probability}:growth_rate={growth_rate}'
  if os.name == 'nt':  # sanitize file path for Windows
        path = sanitize_path(path)

  if not os.path.exists(path):
    os.makedirs(path)
  
  teams_id = ''
  for team in list(teams.keys()):
    teams_id += team + '-'

  colors = ['green', 'blue', 'orange']
  compare_results(steps, title='Mean steps per episode', colors=colors, filename=f'{path}/{teams_id}:steps.png')
  compare_results(scores, title='Mean score per episode', colors=colors, filename=f'{path}/{teams_id}:scores.png')

# sacrificial lamb: if apples fall below a threshold, the agent the furthest away
# from apples will sacrifice itself to feed the others