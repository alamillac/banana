from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class DQN:
    def __init__(
        self,
        max_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        save_path="checkpoint.pth",
    ):
        """Deep Q-Learning.

        Params
        ======
            max_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.max_t = max_t
        self.max_episodes = max_episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.save_path = save_path

    def plot_scores(self, scores):
        # plot the scores
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()

    def test(self, env, agent, num_episodes=100):
        scores = []
        max_score = -np.Inf
        min_score = np.Inf
        for i_episode in range(num_episodes):
            state = env.reset()
            score = 0
            step = 0
            for _ in range(self.max_t):
                action = agent.act(state)
                state, reward, done = env.step(action)
                score += reward
                print(f"\rStep {step} Action {action} Score: {score}", end="")
                step += 1
                if done:
                    break

            max_score = max(score, max_score)
            min_score = min(score, min_score)
            scores.append(score)
            avg_score = np.mean(scores)
            print(f"\rEpisode {i_episode + 1} Score: {score:.2f} Min Score: {min_score:.2f} Max Score: {max_score:.2f} Average Score: {avg_score:.2f}")
        env.close()
        return scores

    def _train(self, env, agent, print_step=False):
        eps = self.eps_start  # initialize epsilon
        for i_episode in range(1, self.max_episodes + 1):
            state = env.reset()
            score = 0
            step = 0
            for _ in range(self.max_t):
                action = agent.act(state, eps)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if print_step:
                    print(f"\rStep {step} Action {action} Score: {score}", end="")
                step += 1
                if done:
                    break
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            yield i_episode, score

    def train_until(self, env, agent, desired_score, consecutive_episodes=100):
        scores_window = deque(maxlen=consecutive_episodes)  # last scores
        scores = []  # list containing scores from each episode
        for i_episode, score in self._train(env, agent):
            scores.append(score)

            scores_window.append(score)  # save most recent score
            avg_score = np.mean(scores_window)

            if avg_score >= desired_score and i_episode > consecutive_episodes:
                print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}")
                break

            print(f"\rEpisode {i_episode} Average Score: {avg_score:.2f}", end="")

        agent.save(self.save_path)
        env.close()
        return scores

    def train(self, env, agent):
        scores_window = deque(maxlen=100)  # last 100 scores
        scores = []  # list containing scores from each episode
        max_score = -np.Inf
        for i_episode, score in self._train(env, agent, print_step=True):
            scores.append(score)

            scores_window.append(score)  # save most recent score
            avg_score = np.mean(scores_window)

            # Save the model if the agent is improving
            if avg_score > max_score:
                max_score = avg_score
                agent.save(self.save_path)

            print(f"\rEpisode {i_episode} Average Score: {avg_score:.2f} Max avg Score: {max_score:.2f}")

            if i_episode % 10 == 0:
                print(f"Memory size: {len(agent.memory)}")

        agent.save(self.save_path)
        env.close()
        return scores
