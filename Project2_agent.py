import numpy as np


class MCAgent:
    """RL agent for the Monte Carlo Black Jack game"""

    def __init__(self):
        self.number_of_states = 203
        self.number_of_actions = 2
        self.q = np.zeros((self.number_of_states + 1, self.number_of_actions), dtype="float64")
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = 1
        self.alpha = 0.1
        self.gamma = 0.9
        self.current_value = 0
        self.visits = [0] * (self.number_of_states + 1)
        self.trajectory = []

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions

    def e_greedy(self, actions):
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        # Make epsilon decay with time, function of number of episodes
        epsilon_adj = self.epsilon * (1 - (self.turn - 500))
        if epsilon_adj <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state):
        self.turn += 1
        self.state = state
        actions = self.q[state, ]
        action = self.e_greedy(actions)
        self.action = action
        return action

    def update_q(self, new_state, reward):
        self.next_state = new_state
        n = self.visits[self.state]
        self.q[self.state, self.action] = reward + (1 / n) * (self.gamma * max(self.q[new_state, ]) - reward)

    def store_trajectory(self, current_state, action, reward, next_state):
        # Increment number of visits
        self.visits[self.state] = self.visits[self.state] + 1
        # Store trajectory
        if self.current_value >= 12 and self.current_value <= 21:
            state_action_pair = (current_state, action)
            self.trajectory.append(state_action_pair)

    def reset(self):
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.current_value = 0
        self.trajectory = []

    def reset_trajectory(self):
        self.trajectory = []
