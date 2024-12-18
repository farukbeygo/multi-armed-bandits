def FollowTheLeader(bandit, n):
    # Initialize a list to store rewards for each arm
    arm_rewards = [0] * bandit.num_arms
    arm_pulls = [0] * bandit.num_arms  # Count of pulls for each arm

    for t in range(n):
        # Choose the arm with the highest average reward so far
        best_arm = 0
        best_average_reward = float('-inf')

        for i in range(bandit.num_arms):
            # Calculate average reward for arm i if it has been pulled at least once
            if arm_pulls[i] > 0:
                average_reward = arm_rewards[i] / arm_pulls[i]
            else:
                average_reward = 0  # If not pulled yet, treat as 0 reward

            # Update best_arm if this arm has a higher average reward
            if average_reward > best_average_reward:
                best_arm = i
                best_average_reward = average_reward

        # Pull the chosen arm and update rewards
        reward = bandit.pull(best_arm)
        arm_rewards[best_arm] += reward
        arm_pulls[best_arm] += 1

