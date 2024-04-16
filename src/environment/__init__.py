import random
import time
from collections import deque

import numpy as np
from unityagents import UnityEnvironment

MAX_SEED_RANGE = 100000
MAX_RESET_COUNT = 50  # To control memory leak in unity environment :S
GRAY_SCALE = False

MEMORY_SIZE = 3

class Env:
    def __init__(self, file_name, train_mode=True, visual=False, seed=None):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.train_mode = train_mode
        self.visual = visual
        self.reset_count = 0
        if self.visual:
            no_graphics = False
        else:
            no_graphics = self.train_mode

        if seed is None:
            seed = random.randrange(MAX_SEED_RANGE)

        self.env = UnityEnvironment(
            file_name=file_name, no_graphics=no_graphics, seed=seed
        )

        # get the default brain
        self.brain_name = self.env.brain_names[0]

        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self._init_memory(env_info)

        # Number of agents in the environment
        self.num_agents = len(env_info.agents)

        # state space
        if self.visual:
            state = self._get_state(env_info)
            self.state_size = state.shape
        else:
            state = self._get_state(env_info)
            self.state_size = len(state)

        # number of actions
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

    def _get_env_info(self, env_info):
        next_state = self._get_state(env_info)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done

    def _init_memory(self, env_info):
        if not self.visual:
            return

        observation = self._get_visual_observation(env_info)
        for _ in range(MEMORY_SIZE):
            self.memory.append(observation)

    def _to_gray_scale(self, image):
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype('float32') # extract luminance
        return gray[..., np.newaxis]  # Add channel dimension

    def _get_visual_observation(self, env_info):
        image = env_info.visual_observations[0]
        if GRAY_SCALE:
            image_gray = self._to_gray_scale(image)
            return np.transpose(
                image_gray, (0, 3, 1, 2)
            )  # Change Channel position: NHWC -> NCHW

        return np.transpose(
            image, (0, 3, 1, 2)
        )  # Change Channel position: NHWC -> NCHW

    def _get_visual_state(self, memory):
        # TODO: Refactor this
        return np.concatenate(memory, axis=1)

    def _get_state(self, env_info):
        if self.visual:
            return self._get_visual_state(self.memory)
        return env_info.vector_observations[0]

    def reset(self):
        self.reset_count += 1

        if self.visual and self.reset_count % (MAX_RESET_COUNT + 1) == 0:
            # Do a restart every MAX_RESET_COUNT to avoid memory leak
            env_info = self.env.restart(train_mode=self.train_mode)[self.brain_name]
        else:
            env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

        # Clear memory
        self.memory.clear()
        self._init_memory(env_info)

        return self._get_state(env_info)

    def _update_memory(self, env_info):
        if not self.visual:
            return

        observation = self._get_visual_observation(env_info)
        self.memory.append(observation)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        if not self.train_mode:
            time.sleep(0.1)

        self._update_memory(env_info)
        return self._get_env_info(env_info)

    def close(self):
        self.env.close()
