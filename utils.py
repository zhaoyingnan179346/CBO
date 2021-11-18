import numpy as np
import torch
import random


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

    def load_part_date(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        state_buffer = np.load(f"{save_folder}_state.npy")
        action_buffer = np.load(f"{save_folder}_action.npy")
        next_state_buffer = np.load(f"{save_folder}_next_state.npy")
        not_done_buffer = np.load(f"{save_folder}_not_done.npy")
        count = 0
        # random_statex1 = [1.67, 2.06]
        # random_statey1 = [1.43, 1.88]
        # random_statex2 = [3.66, 4.00]
        # random_statey2 = [3.99, 4.54]
        # random_statex3 = [2.65, 3.04]
        # random_statey3 = [3.58, 3.89]

        buffer_index = 0
        temp_size = min(reward_buffer.shape[0], self.max_size)
        print('temp_size is:')
        print(temp_size)
        random_state_list = []
        num_drop = 20
        range_x = 0.5
        range_y = 0.5
        for i in range(num_drop):
            random_index = np.random.randint(0, temp_size)
            select_state = state_buffer[random_index]
            temp_state_x = select_state[0] + range_x
            temp_state_y = select_state[1] + range_y
            random_state_list.append([select_state[0], select_state[1], temp_state_x, temp_state_y])

        # rand_indices = np.random.permutation(temp_size)[:int(temp_size * 0.5]
        for i in range(temp_size):
            state = state_buffer[i]
            in_drop_out = False
            for temp_point in random_state_list:
                state1_x, state1_y, state2_x, state2_y = temp_point
                if (state[0] >= state1_x and state[0] <= state2_x) and (state[1] >= state1_y and state[1] <= state2_y):
                    in_drop_out = True
                    break

            if not in_drop_out:
                self.state[buffer_index] = state_buffer[i]
                self.action[buffer_index] = action_buffer[i]
                self.next_state[buffer_index] = next_state_buffer[i]
                self.reward[buffer_index] = reward_buffer[i]
                self.not_done[buffer_index] = not_done_buffer[i]
                buffer_index += 1

        self.size = buffer_index - 1

    def save_part_data(self, save_folder):
        print('buffer size ' + str(self.size))
        np.save(f"{save_folder}_part_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_part_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_part_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_part_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_part_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_part_ptr.npy", self.ptr)