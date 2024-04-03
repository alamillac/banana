from agent import Agent, DQN
from environment import Env

env = Env(train_mode=False)

dqn = DQN()
agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)
agent.load("checkpoint.pth")

dqn.test(env, agent)
