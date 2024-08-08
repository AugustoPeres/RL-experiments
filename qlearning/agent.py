import random

import torch
import torch.optim as optim


class QLearningAgent():

    def __init__(self, network, batch_size, gamma, epsilon, learning_rate,
                 number_of_actions):
        """Args:
        network: Neural network to be used for training
        """
        self.nn = network
        self.target_network = copy.deepcopy(network)

        self.experience_replay = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.number_actions = number_of_actions

        self.optimizer = optim.AdamW(self.nn.parameters(),
                                     lr=learning_rate,
                                     amsgrad=True)

    def policy(self, observation):
        if random.gauss() > self.epsilon:
            return np.random.choice(range(self.number_actions))
        with torch.no_grad():
            q_values = self.nn(observation).squeeze(-1)
        action = torch.argmax(q_values)
        return action.numpy()

    def collect_observation(self, observation, action, reward,
                            next_observation, is_terminal):
        self.experience_replay.append(
            (observation, action, reward, next_observation, is_terminal))

    def update(self):
        training_batch = self.get_random_batch_from_replay_buffer()
        training_inputs = self.make_batched_input(training_batch)
        training_targets, masks = self.make_batched_target(training_batch)

        pred_q_values = self.nn(training_inputs)
        loss = torch.nn.MSELoss()(pred_q_values * masks, training_targets)
        self.optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def make_batched_input(self, training_batch):
        states = [x[0] for x in training_batch]
        return torch.stack(states)

    def make_batched_target(self, training_batch):
        targets = []
        masks = []
        for (observation, action, reward, next_observation,
             is_terminal) in training_batch:
            mask = torch.eye(self.number_of_actions)[action]
            if is_terminal:
                targets.append(mask * reward)
            else:
                with torch.no_grad():
                    preds = self.target_network(observation.unsqueeze(0))
                max_q_value = torch.max(preds)
                targets.append(mask * reward + self.gamma * mask * max_q_value)
            masks.append(mask)
        return torch.stack(targets), torch.stack(masks)

    def get_random_batch_from_replay_buffer(self):
        return random.sample(self.experience_replay, self.batch_size)

    def update_target_network(self):
        self.target_network.load_state_dict(self.nn)
