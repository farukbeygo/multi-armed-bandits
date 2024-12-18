import numpy as np

class UCBBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms  # number of arms
        self.counts = np.zeros(n_arms)  # times each arm was pulled
        self.values = np.zeros(n_arms)  # estimated value of each arm

    def select_arm(self, round):
        # Calculate UCB for each arm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                # Select untried arms to encourage exploration
                return arm
            # Calculate the UCB for each arm using Hoeffding's bound
            average_reward = self.values[arm]
            confidence_bound = np.sqrt((2 * np.log(round + 1)) / self.counts[arm])
            ucb_values[arm] = average_reward + confidence_bound

        # Choose the arm with the highest UCB
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        # Increment the count of the chosen arm
        self.counts[chosen_arm] += 1
        # Update the running average reward for the chosen arm
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update formula to keep running average
        new_value = ((n - 1) * value + reward) / n
        self.values[chosen_arm] = new_value

# Example usage
n_arms = 5  # Number of arms
n_rounds = 100  # Number of rounds
bandit = UCBBandit(n_arms)

# Simulate rewards for each arm (for testing purposes)
# Each arm has a fixed probability of giving a reward
true_rewards = np.random.rand(n_arms)

for round in range(n_rounds):
    # Select an arm based on UCB
    chosen_arm = bandit.select_arm(round)
    # Simulate a reward for the chosen arm (1 for reward, 0 for no reward)
    reward = 1 if np.random.rand() < true_rewards[chosen_arm] else 0
    # Update the bandit with the observed reward
    bandit.update(chosen_arm, reward)

# Results
print("Estimated values for each arm:", bandit.values)
print("Number of times each arm was pulled:", bandit.counts)
