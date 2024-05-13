from gym import Wrapper

class SingleAgentWrapper(Wrapper):

  """
  A Wrapper for Multi-Agent environments which only contain a single agent.
  """

  def __init__(self, env, agent_id):
    super(SingleAgentWrapper, self).__init__(env)
    assert env.n_agents == 1
    self.agent_id = agent_id
