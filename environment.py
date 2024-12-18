import numpy as np

class Environment:
    """
    A class representing the environment for a multi-armed bandit problem.

    Attributes:
        reward_probabilities (list[float]): The probability of receiving a reward for each arm.
        actual_rewards (list[float]): The actual reward values associated with each arm.
        k_arms (int): The number of arms in the environment.
    """

    def __init__(self, reward_probabilities, actual_rewards):
        """
        Initializes the Environment instance.

        Args:
            reward_probabilities (list[float]): Probabilities of receiving a reward for each arm.
            actual_rewards (list[float]): Actual reward values for each arm.

        Raises:
            ValueError: If the lengths of reward_probabilities and actual_rewards do not match.
        """
        if len(reward_probabilities) != len(actual_rewards):
            raise ValueError(
                f"Size of reward probabilities ({len(reward_probabilities)}) does not match size of actual rewards ({len(actual_rewards)})."
            )

        if not all(0 <= p <= 1 for p in reward_probabilities):
            raise ValueError("All reward probabilities must be in the range [0, 1].")

        self.reward_probabilities = reward_probabilities
        self.actual_rewards = actual_rewards
        self.k_arms = len(reward_probabilities)

    def choose_arm(self, arm):
        """
        Simulates choosing an arm and returns the reward.

        Args:
            arm (int): The index of the arm to choose (0-based).

        Returns:
            float: The reward obtained from the chosen arm (0.0 if no reward).

        Raises:
            ValueError: If the specified arm index is out of bounds.
        """
        if not (0 <= arm < self.k_arms):
            raise ValueError(f"Arm index must be between 0 and {self.k_arms - 1}, but got {arm}.")

        reward_received = np.random.random() < self.reward_probabilities[arm]
        return self.actual_rewards[arm] if reward_received else 0.0

    def __str__(self):
        """
        Returns a string representation of the environment.

        Returns:
            str: A summary of the environment's parameters.
        """
        return (
            f"Environment with {self.k_arms} arms\n"
            f"Reward Probabilities: {self.reward_probabilities}\n"
            f"Actual Rewards: {self.actual_rewards}"
        )