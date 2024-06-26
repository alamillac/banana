import numpy as np
from unityagents import UnityEnvironment
import time
import matplotlib.pyplot as plt

def to_gray_scale(image):
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype('float32') # extract luminance
    return gray[..., np.newaxis]  # Add channel dimension

# please do not modify the line below
env = UnityEnvironment(file_name="./VisualBanana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print("Number of agents:", len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print("Number of actions:", action_size)

# examine the state space
state = env_info.visual_observations[0]
print('States look like:')
plt.imshow(np.squeeze(state))
plt.show()
state_size = state.shape
print('States have shape:', state.shape)

# Gray scale state
state = to_gray_scale(env_info.visual_observations[0])
print('Gray scale states look like:')
plt.imshow(np.squeeze(state))
plt.show()
state_size = state.shape
print('Gray scale states have shape:', state.shape)

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.visual_observations[0]  # get the current state
score = 0  # initialize the score
step = 0
while True:
    action = np.random.randint(action_size)  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.visual_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    step += 1
    time.sleep(0.1)
    #print(f'\rStep {step} Score: {score}', end="")
    print(f'Step {step} Score: {score}')
    if done:  # exit loop if episode finished
        break

env.close()
