from agent import Agent, DQN
from environment import Env

env = Env(train_mode=True)

print("Number of agents:", env.num_agents)

# number of actions
print("Number of actions:", env.action_size)

# examine the state space
print("States have length:", env.state_size)

dqn = DQN(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_path="checkpoint.pth")
agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)
agent.load("checkpoint.pth") ## Move to train ??

dqn.train(env, agent)
