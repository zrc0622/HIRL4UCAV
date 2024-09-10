import numpy as np
import pandas as pd

def read_and_split_data_with_rewards(data_dir, output_dir, return_calc):
    """
    Load data from a CSV file, convert it into structured numpy arrays for states and actions,
    split the data into multiple trajectories based on the state values, and calculate the total
    rewards for each trajectory using the provided reward calculation function.
    
    Parameters:
    data_dir (str): The path to the CSV file containing the data.
    output_dir (str): The directory to save the split trajectories and their rewards.
    return_calc (function): A function that takes a state (14-dimensional) and an action
                            (4-dimensional) and returns the reward for that state-action pair.
    """
    # Load the CSV data
    expert_data = pd.read_csv(data_dir, header=None)
    
    # Extract and convert the state data
    state = expert_data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ') for item in state])
    
    # Extract and convert the action data
    action = expert_data.iloc[1].to_numpy()
    npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
    
    # Prepare to store the indices where trajectories should be split
    split_indices = []
    
    # Assume npstate shape is (n, 14) where n is the number of columns (transposed for processing)
    for i in range(10, len(npstate) - 10):  # Ensuring there is room to check both before and after
        if np.all(npstate[i-5:i, -1] == -1) and np.all(npstate[i:i+5, -1] == 1):
            split_indices.append(i)
    
    # Split the state and action arrays using the found indices
    state_trajectories = np.split(npstate, split_indices, axis=0)
    action_trajectories = np.split(npaction, split_indices, axis=0)
    
    # Calculate rewards for each trajectory
    rewards = []
    for traj_state, traj_action in zip(state_trajectories, action_trajectories):
        total_reward = sum(return_calc(s, a) for s, a in zip(traj_state, traj_action))
        rewards.append(total_reward)

    # Save each trajectory and their total reward to a separate CSV file
    for index, (traj_state, traj_action, reward) in enumerate(zip(state_trajectories, action_trajectories, rewards)):
        # state_file_path = f"{output_dir}/state_trajectory_{index}.csv"
        # action_file_path = f"{output_dir}/action_trajectory_{index}.csv"
        # pd.DataFrame(traj_state).to_csv(state_file_path, header=False, index=False)
        # pd.DataFrame(traj_action).to_csv(action_file_path, header=False, index=False)
        print(f"Trajectory {index}: Total Reward = {reward}")
    
    # print(f"Saved {len(state_trajectories)} trajectories and their rewards to {output_dir}.")
    print(f"Average Reward = {sum(rewards)/len(rewards)}")

def calc(state, action):
    # 距离奖励
    distance = -np.sqrt(np.sum(np.square(state[:3]))*1e8)*0.0001
    
    # 锁敌奖励
    lock = -10*state[10]
    
    # 开火奖励
    if action[-1] > 0:
        if state[-1] > 0.5 and state[-2] < -0.5:
            fire = -4
        elif state[-1] > 0.5 and state[-2] > 0.5:
            fire = 100
        else: fire = -1
    else: fire = 0

    return distance+lock+fire

# Example usage:
read_and_split_data_with_rewards('B:/code/C Primer Plus/test_py/expert_data_bc1.csv', 'B:/code/C Primer Plus/test_py', calc)
