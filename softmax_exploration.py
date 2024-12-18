import numpy as np


class SoftmaxAgent(object):

    def __init__(self, environment, max_iterations=500, tau=0.15):
        self.environment = environment
        self.iterations = max_iterations
        self.tau = tau

        self.action_space = np.zeros(self.environment.k_arms)
        self.q_values = np.zeros(self.environment.k_arms)
        self.arm_counts = np.zeros(self.environment.k_arms)
        self.arm_rewards = np.zeros(self.environment.k_arms)

        self.rewards = [0.0]
        self.cum_rewards = [0.0]

    def act(self):
        for i in range(self.iterations):
            self.action_space = np.exp(self.q_values/self.tau) / np.sum(np.exp(self.q_values/self.tau))

            arm = np.random.choice(self.environment.k_arms, p=self.action_space)
            reward = self.environment.choose_arm(arm)

            self.arm_counts[arm] = self.arm_counts[arm] + 1
            self.arm_rewards[arm] = self.arm_rewards[arm] + reward

            self.q_values[arm] = self.q_values[arm] + (1/self.arm_counts[arm]) * (reward - self.q_values[arm])

            self.rewards.append(reward)
            self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

        return {"arm_counts": self.arm_counts, "rewards": self.rewards, "cum_rewards": self.cum_rewards}

