from agent import Agent, DQN
from environment import Env

visual = True
if visual:
    save_path = "checkpoint_visual.pth"
else:
    save_path = "checkpoint.pth"

env = Env(train_mode=True, visual=visual)

print("Number of agents:", env.num_agents)
print("Number of actions:", env.action_size)
print("States have length:", env.state_size)

dqn = DQN(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_path=save_path)
agent = Agent(state_size=env.state_size, action_size=env.action_size, visual=visual, seed=0)
#agent.load(save_path) ## Move to train ??

scores = dqn.train(env, agent)
dqn.plot_scores(scores)
