import time
import numpy as np
import random

from unityagents import UnityEnvironment

MAX_SEED_RANGE = 100000
MAX_RESET_COUNT = 50 # To control memory leak in unity environment :S

class Env:
    def __init__(self, train_mode=True, visual=False, seed=None):
        self.train_mode = train_mode
        self.visual = visual
        self.reset_count = 0
        if self.visual:
            file_name = "./VisualBanana_Linux/Banana.x86_64"
            no_graphics = False
        else:
            file_name = "./Banana_Linux/Banana.x86_64"
            no_graphics = self.train_mode

        if seed is None:
            seed = random.randrange(MAX_SEED_RANGE)

        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics, seed=seed)

        # get the default brain
        self.brain_name = self.env.brain_names[0]

        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

        # Number of agents in the environment
        self.num_agents = len(env_info.agents)

        # state space
        if self.visual:
            state = env_info.visual_observations[0]
            sh = state.shape
            self.state_size = (sh[0], sh[3], sh[1], sh[2]) # Change Channel position: NHWC -> NCHW
        else:
            state = env_info.vector_observations[0]
            self.state_size = len(state)

        # number of actions
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

    def _get_env_info(self, env_info):
        if self.visual:
            next_state = self._get_visual_state(env_info)
        else:
            next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done

    def _get_visual_state(self, env_info):
        image = env_info.visual_observations[0]
        return np.transpose(image, (0, 3, 1, 2)) # Change Channel position: NHWC -> NCHW

    def reset(self):
        self.reset_count += 1

        if self.visual and self.reset_count % MAX_RESET_COUNT == 0:
            # Do a restart every MAX_RESET_COUNT to avoid memory leak
            env_info = self.env.restart(train_mode=self.train_mode)[self.brain_name]
        else:
            env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

        if self.visual:
            state = self._get_visual_state(env_info)
        else:
            state = env_info.vector_observations[0]

        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        if not self.train_mode:
            time.sleep(0.1)
        return self._get_env_info(env_info)

    def close(self):
        self.env.close()
