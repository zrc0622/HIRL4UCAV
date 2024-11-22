def get_loc_diff(state):
    loc_diff = ((((state[0]) * 10000) ** 2) + (((state[1]) * 10000) ** 2) + (((state[2]) * 10000) ** 2)) ** (1 / 2)
    return loc_diff


def get_termination(state):
    done = False
    if state[-1] <= 0.1: # 敌机血量低于0则结束
        done = True
    return done

def get_reward(state, action, n_state):
    reward = 0
    step_success = 0
    loc_diff = get_loc_diff(n_state)  # get location difference information for reward
    
    # 距离惩罚：帮助追击
    reward -= (0.0001 * loc_diff)

    # 目标角惩罚：帮助锁敌
    reward -= (n_state[6]) * 10

    # 开火奖励：帮助开火
    if action[-1] > 0: # 如果导弹发射
        reward -= 8
        if state[8] > 0 and state[7] < 0: # 且导弹存在、不锁敌
            # reward -= 100 # 4
            step_success = -1
        elif state[8] > 0 and state[7] > 0: # 且导弹存在、锁敌
            # reward += 8 # 100
            step_success = 1
        else:
            # reward -= 10 # 1
            reward -= 0

    if n_state[-1] < 0.1:
        reward += 600

    return reward, step_success


from hirl.utils.data_processor import read_data
import statistics

# data_dir = './straight_line/expert_data_ai_random.csv'
# data_dir = './serpentine/expert_data_ai_random.csv'
data_dir = './circular/expert_data_ai_random.csv'
expert_states, expert_actions = read_data(data_dir)

i = 0
count = 0
total_return = 0
total_success = 0
return_list = []
success_list = []
while i < expert_states.shape[0]:
    state = expert_states[i]
    action = expert_actions[i]
    next_state = expert_states[i+1]

    reward, step_success = get_reward(state, action, next_state)
    total_return += reward
    total_success += step_success

    done = get_termination(next_state)
    
    if done: 
        i += 1
        count += 1

        return_list.append(total_return)
        total_return = 0
        success_list.append(total_success)
        total_success = 0

    i += 1

print(statistics.mean(return_list), statistics.stdev(return_list))
print(statistics.mean(success_list), statistics.stdev(success_list))