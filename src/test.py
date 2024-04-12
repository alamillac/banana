from agent import Agent, DQN
from environment import Env

visual = True
if visual:
    save_path = "checkpoint_visual.pth"
else:
    save_path = "checkpoint.pth"

env = Env(train_mode=True, visual=visual, seed=0)

dqn = DQN()
agent = Agent(state_size=env.state_size, action_size=env.action_size, visual=visual, seed=0)
agent.load(save_path)

dqn.test(env, agent, num_episodes=100)
