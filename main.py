from environment import Environment
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for consistency
sns.set_theme(style="whitegrid")

class UCBAgent:
    """
    Implements the Upper Confidence Bound (UCB) algorithm for the multi-armed bandit problem.
    """

    def __init__(self, environment, max_iterations=500, c=1.0):
        self.environment = environment
        self.iterations = max_iterations
        self.c = c
        self.q_values = np.zeros(self.environment.k_arms)
        self.arm_counts = np.zeros(self.environment.k_arms)
        self.rewards = [0.0]
        self.cum_rewards = [0.0]

    def act(self):
        """
        Executes the UCB algorithm over the specified number of iterations.
        Returns:
            dict: Contains arm counts, rewards, and cumulative rewards.
        """
        for t in range(1, self.iterations + 1):
            # Select arm based on UCB formula
            if t <= self.environment.k_arms:
                arm = t - 1  # Pull each arm once initially
            else:
                ucb_values = self.q_values + self.c * np.sqrt(np.log(t) / (self.arm_counts + 1e-10))
                arm = np.argmax(ucb_values)

            # Get reward and update statistics
            reward = self.environment.choose_arm(arm)
            self.arm_counts[arm] += 1
            self.q_values[arm] += (1 / self.arm_counts[arm]) * (reward - self.q_values[arm])
            self.rewards.append(reward)
            self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

        return {
            "arm_counts": self.arm_counts,
            "rewards": self.rewards,
            "cum_rewards": self.cum_rewards,
        }

# Define the environment
reward_probs = [0.6, 0.08, 0.19, 0.57, 0.18, 0.78, 0.11, 0.82, 0.22, 0.77, 0.31, 0.62, 0.6, 0.75, 0.09, 0.33, 0.86, 0.58, 0.66, 0.26,
                0.57, 0.79, 0.83, 0.83, 0.43, 0.72, 0.71, 0.34, 0.53, 0.19, 0.35, 0.69, 0.36, 0.68, 0.91, 0.71, 0.1, 0.57, 0.1, 0.39,
                0.67, 0.61, 0.31, 0.5, 0.31, 0.87, 0.61, 0.53, 0.92, 0.67, 0.24, 0.09, 0.22, 0.88, 0.12, 0.55, 0.91, 0.46, 0.19, 0.62,
                0.45, 0.03, 0.79, 0.58, 0.71, 0.23, 0.2, 0.28, 0.24, 0.46, 0.48, 0.67, 0.07, 0.54, 0.78, 0.89, 0.67, 0.57, 0.05, 0.33,
                0.57, 0.72, 0.61, 0.82, 0.07, 0.31, 0.42, 0.37, 0.82, 0.96, 1.0, 0.72, 0.08, 1.0, 0.38, 0.23, 0.69, 0.85, 0.0, 0.05]
actual_probs = [1.0] * 100
test_env = Environment(reward_probabilities=reward_probs, actual_rewards=actual_probs)

# Parameters
iterations = 10000
epsilon = 0.1
ucb_c = 2.0

# Epsilon-Greedy Agent
eg_agent = EpsilonGreedyAgent(test_env, max_iterations=iterations, epsilon=epsilon)
eg_result = eg_agent.act()

# UCB Agent
ucb_agent = UCBAgent(test_env, max_iterations=iterations, c=ucb_c)
ucb_result = ucb_agent.act()

# Plot cumulative rewards
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(len(eg_result["cum_rewards"])), y=eg_result["cum_rewards"], label="Epsilon-Greedy", color='blue')
sns.lineplot(x=np.arange(len(ucb_result["cum_rewards"])), y=ucb_result["cum_rewards"], label="UCB", color='green')
plt.title("Cumulative Rewards Over Iterations", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Cumulative Rewards", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot arm counts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Epsilon-Greedy Arm Counts
sns.barplot(x=np.arange(len(eg_result["arm_counts"])), y=eg_result["arm_counts"], ax=ax1, color='blue', alpha=0.7)
ax1.set_title("Epsilon-Greedy: Number of Times Each Arm Was Pulled", fontsize=16)
ax1.set_xlabel("Arm Index", fontsize=14)
ax1.set_ylabel("Pull Count", fontsize=14)

# Adjust x-axis labels for Epsilon-Greedy
step_eg = max(1, len(eg_result["arm_counts"]) // 10)  # Show every 10th arm index
ax1.set_xticks(np.arange(0, len(eg_result["arm_counts"]), step_eg))
ax1.set_xticklabels([f"Arm {i}" for i in np.arange(0, len(eg_result["arm_counts"]), step_eg)], rotation=45, ha='right')

# UCB Arm Counts
sns.barplot(x=np.arange(len(ucb_result["arm_counts"])), y=ucb_result["arm_counts"], ax=ax2, color='green', alpha=0.7)
ax2.set_title("UCB: Number of Times Each Arm Was Pulled", fontsize=16)
ax2.set_xlabel("Arm Index", fontsize=14)
ax2.set_ylabel("Pull Count", fontsize=14)

# Adjust x-axis labels for UCB
step_ucb = max(1, len(ucb_result["arm_counts"]) // 10)  # Show every 10th arm index
ax2.set_xticks(np.arange(0, len(ucb_result["arm_counts"]), step_ucb))
ax2.set_xticklabels([f"Arm {i}" for i in np.arange(0, len(ucb_result["arm_counts"]), step_ucb)], rotation=45, ha='right')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
