from agent import Agent, DQN
from environment import Env

visual = False
if visual:
    save_path = "checkpoint_visual.pth"
else:
    save_path = "checkpoint.pth"

env = Env(train_mode=False, visual=visual)

dqn = DQN()
agent = Agent(state_size=env.state_size, action_size=env.action_size, visual=visual, seed=0)
agent.load(save_path)

dqn.test(env, agent)
