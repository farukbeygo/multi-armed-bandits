import random


class BernoulliBandit:
    def __init__(self, means):
        """
        Initialize the bandit with a list of means, where each mean represents the probability of
        getting a reward of 1 for that arm. Assumes means is a list of K >= 2 floats in [0, 1].
        """
        self.means = means  # Probabilities of reward for each arm
        self.num_arms = len(means)  # Number of arms
        self.total_pulls = [0] * self.num_arms  # Total pulls per arm
        self.total_rewards = [0] * self.num_arms  # Total rewards received per arm
        self.best_mean = max(means)  # Optimal arm mean for regret calculation

    def K(self):
        """
        Return the number of arms.
        """
        return self.num_arms

    def pull(self, a):
        """
        Simulate pulling arm 'a', which returns 1 with probability equal to the mean of the (a+1)th arm.
        Updates the number of pulls and total rewards for the pulled arm.
        """
        reward = 1 if random.random() < self.means[a] else 0
        self.total_pulls[a] += 1
        self.total_rewards[a] += reward
        return reward

    def regret(self):
        """
        Calculate the regret incurred so far. Regret is defined as the difference between the
        reward from always pulling the optimal arm and the actual reward received.
        """
        optimal_reward = self.best_mean * sum(self.total_pulls)
        actual_reward = sum(self.total_rewards)
        return optimal_reward - actual_reward
