from agent import Agent, DQN
from environment import Env

visual = False
if visual:
    save_path = "checkpoint_visual.pth"
    env_filename = "./VisualBanana_Linux/Banana.x86_64"
else:
    save_path = "checkpoint.pth"
    env_filename = "./Banana_Linux/Banana.x86_64"

env = Env(env_filename, train_mode=False, visual=visual, seed=0)

dqn = DQN()
agent = Agent(state_size=env.state_size, action_size=env.action_size, visual=visual, seed=0)
agent.load(save_path)

dqn.test(env, agent, num_episodes=2)
