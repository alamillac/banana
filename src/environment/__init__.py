import time

from unityagents import UnityEnvironment


class Env:
    def __init__(self, train_mode=True):
        self.train_mode = train_mode

        if self.train_mode:
            self.env = UnityEnvironment(
                file_name="./Banana_Linux/Banana.x86_64", no_graphics=True
            )
        else:
            self.env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

        # get the default brain
        self.brain_name = self.env.brain_names[0]

        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # Number of agents in the environment
        self.num_agents = len(env_info.agents)

        # state space
        state = env_info.vector_observations[0]
        self.state_size = len(state)

        # number of actions
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

    def _get_env_info(self, env_info):
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        if not self.train_mode:
            time.sleep(0.1)
        return self._get_env_info(env_info)

    def close(self):
        self.env.close()
