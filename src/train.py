from agent import Agent, DQN
from environment import Env

visual = False
if visual:
    save_path = "model_visual.pth"
    save_checkpoint_path = "checkpoint_visual.pth"
    env_filename = "./VisualBanana_Linux/Banana.x86_64"
else:
    save_path = "model.pth"
    save_checkpoint_path = "checkpoint.pth"
    env_filename = "./Banana_Linux/Banana.x86_64"

env = Env(env_filename, train_mode=True, visual=visual)

print("Number of agents:", env.num_agents)
print("Number of actions:", env.action_size)
print("States have length:", env.state_size)

dqn = DQN(max_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_model_path=save_path, save_checkpoint_path=save_checkpoint_path)

agent = Agent(state_size=env.state_size, action_size=env.action_size, visual=visual, seed=0)

#scores = dqn.train_until(env, agent, desired_score=13, consecutive_episodes=100)
scores = dqn.train(env, agent)
dqn.plot_scores(scores)
