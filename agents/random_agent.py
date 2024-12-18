import numpy as np


class RandomAgent:
    """
    A RandomAgent that selects arms at random in a multi-armed bandit environment.

    Attributes:
        environment: The environment representing the multi-armed bandit.
        iterations: The maximum number of iterations the agent will perform.
        q_values: Estimated Q-values (expected rewards) for each arm.
        arm_counts: Count of how many times each arm has been pulled.
        arm_rewards: Total rewards collected from each arm.
        rewards: List of rewards obtained at each iteration.
        cumulative_rewards: List of cumulative average rewards up to each iteration.
    """

    def __init__(self, environment, max_iterations=500):
        """
        Initializes the RandomAgent.

        Args:
            environment: An object representing the multi-armed bandit environment.
            max_iterations (int): The maximum number of iterations the agent will perform.
        """
        self.environment = environment
        self.iterations = max_iterations

        # Initialize statistics for the agent
        self.q_values = np.zeros(self.environment.k_arms)
        self.arm_counts = np.zeros(self.environment.k_arms)
        self.arm_rewards = np.zeros(self.environment.k_arms)

        # Track rewards
        self.rewards = [0.0]
        self.cumulative_rewards = [0.0]

    def act(self):
        """
        Performs actions in the environment by selecting arms at random.

        The agent updates its estimates of Q-values, counts of arm pulls,
        and tracks rewards at each iteration.

        Returns:
            dict: A dictionary containing:
                - "arm_counts": The count of pulls for each arm.
                - "rewards": The list of rewards obtained.
                - "cumulative_rewards": The list of cumulative average rewards.
        """
        for iteration in range(self.iterations):
            # Randomly select an arm
            arm = np.random.choice(self.environment.k_arms)

            # Get reward for the selected arm
            reward = self.environment.choose_arm(arm)

            # Update statistics for the selected arm
            self.arm_counts[arm] += 1
            self.arm_rewards[arm] += reward

            # Update Q-value using incremental formula
            self.q_values[arm] += (1 / self.arm_counts[arm]) * (reward - self.q_values[arm])

            # Update reward tracking
            self.rewards.append(reward)
            cumulative_average_reward = sum(self.rewards) / len(self.rewards)
            self.cumulative_rewards.append(cumulative_average_reward)

        return {
            "arm_counts": self.arm_counts,
            "rewards": self.rewards,
            "cumulative_rewards": self.cumulative_rewards,
        }
