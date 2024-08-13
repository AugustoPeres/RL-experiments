import copy
import random

import torch
import torch.optim as optim


class QLearningAgent():

    def __init__(self,
                 network,
                 batch_size,
                 gamma,
                 start_epsilon,
                 min_epsilon,
                 decay,
                 learning_rate,
                 number_of_actions,
                 target_network_update_frequency,
                 max_steps_in_buffer=int(1e4)):
        """Args:

        network: Neural network to be used for training.
        batch_size: Sizes of the batch to be samples from the replay buffer.
        gamma: Future Reward importance.
        start_epsilon: Start exploration rate.
        min_epsilon: Min epsilon we allow to decay to.
        decay: Speed of the exploration rate (epsilon) decay.
        learning_rate: Learning rate to be used by the optimizer.
        number_of_actions: Number of actions available to the agent.
        target_network_update_frequency: Frequency to copy params to target
          network.
        max_steps_in_buffer: Maximum examples to store in the replay buffer.
        """
        self.nn = network
        self.target_network = copy.deepcopy(network)
        self.target_network_update_frequency = target_network_update_frequency
        self.steps_since_target_network_update = 1

        self.experience_replay = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.number_of_actions = number_of_actions

        self.optimizer = optim.AdamW(self.nn.parameters(),
                                     lr=learning_rate,
                                     amsgrad=True)

        self.max_steps_in_buffer = max_steps_in_buffer

    def policy(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            return random.sample(range(self.number_of_actions), 1)[0]
        with torch.no_grad():
            q_values = self.nn(
                torch.tensor(observation).unsqueeze(0)).squeeze(-1)
        action = torch.argmax(q_values)
        return action.numpy()

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def collect_observation(self, observation, action, reward,
                            next_observation, is_terminal):
        self.experience_replay.append(
            (observation, action, reward, next_observation, is_terminal))
        if len(self.experience_replay) > self.max_steps_in_buffer:
            self.experience_replay = self.experience_replay[
                -self.max_steps_in_buffer:]

    def update(self):
        if len(self.experience_replay) < self.batch_size:
            return
        training_batch = self.get_random_batch_from_replay_buffer()
        training_inputs = self.make_batched_input(training_batch)
        training_targets, masks = self.make_batched_target(training_batch)

        pred_q_values = self.nn(training_inputs)
        loss = torch.nn.MSELoss()(pred_q_values * masks, training_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_batched_input(self, training_batch):
        states = [torch.tensor(x[0]) for x in training_batch]
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
                    preds = self.target_network(
                        torch.tensor(next_observation).unsqueeze(0))
                max_q_value = torch.max(preds)
                targets.append(mask * (reward + self.gamma * max_q_value))
            masks.append(mask)
        return torch.stack(targets), torch.stack(masks)

    def get_random_batch_from_replay_buffer(self):
        return random.sample(self.experience_replay, self.batch_size)

    def update_target_network(self):
        if self.steps_since_target_network_update % self.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.nn.state_dict())
            self.steps_since_target_network_update = 0
        self.steps_since_target_network_update += 1
