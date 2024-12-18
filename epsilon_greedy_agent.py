import numpy as np


class EpsilonGreedyAgent:
    """
    An Epsilon-Greedy agent for the multi-armed bandit problem.

    Attributes:
        environment: The environment representing the multi-armed bandit.
        iterations: The maximum number of iterations the agent will perform.
        epsilon: The probability of exploring a random arm.
        q_values: Estimated Q-values (expected rewards) for each arm.
        arm_counts: Count of how many times each arm has been pulled.
        arm_rewards: Total rewards collected from each arm.
        rewards: List of rewards obtained at each iteration.
        cum_rewards: List of cumulative average rewards up to each iteration.
    """

    def __init__(self, environment, max_iterations=500, epsilon=0.1):
        """
        Initializes the EpsilonGreedyAgent.

        Args:
            environment: An object representing the multi-armed bandit environment.
            max_iterations (int): The maximum number of iterations the agent will perform.
            epsilon (float): The initial probability of exploring a random arm.
        """
        self.environment = environment
        self.iterations = max_iterations
        self.epsilon = epsilon

        # Initialize statistics for the agent
        self.q_values = np.zeros(self.environment.k_arms)
        self.arm_counts = np.zeros(self.environment.k_arms)
        self.arm_rewards = np.zeros(self.environment.k_arms)

        # Track rewards
        self.rewards = [0.0]
        self.cum_rewards = [0.0]

    def act(self):
        """
        Performs actions in the environment using the epsilon-greedy strategy.

        The agent selects an arm based on epsilon-greedy exploration-exploitation:
        - With probability epsilon, it selects a random arm.
        - Otherwise, it selects the arm with the highest estimated Q-value.

        Returns:
            dict: A dictionary containing:
                - "arm_counts": The count of pulls for each arm.
                - "rewards": The list of rewards obtained.
                - "cum_rewards": The list of cumulative average rewards.
        """
        for iteration in range(self.iterations):
            # Choose an arm based on epsilon-greedy strategy
            if np.random.random() < self.epsilon:
                arm = np.random.choice(self.environment.k_arms)  # Explore
            else:
                arm = np.argmax(self.q_values)  # Exploit

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
            self.cum_rewards.append(cumulative_average_reward)

        return {
            "arm_counts": self.arm_counts,
            "rewards": self.rewards,
            "cum_rewards": self.cum_rewards,
        }
