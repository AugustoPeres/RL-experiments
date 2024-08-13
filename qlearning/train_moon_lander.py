"""Script to train the moonlander"""
import argparse

import matplotlib.pyplot as plt
import torch
import gymnasium as gym

from agent import QLearningAgent


def record_video(agent, name_prefix):
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env,
                                   video_folder='output_videos',
                                   name_prefix=name_prefix)
    observation, _ = env.reset()
    env.start_video_recorder()
    terminated = False
    truncated = False

    while (not terminated) and (not truncated):
        action = agent.policy(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()

    env.close_video_recorder()
    env.close()


def train_agent(number_of_episodes, agent, record_evey_n_episodes=100):
    env = gym.make('LunarLander-v2')
    cummulative_rewards = []
    for e in range(number_of_episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        cummulative_reward = 0

        if e % record_evey_n_episodes == 0:
            record_video(agent, f'episode_{e}')

        while (not terminated) and (not truncated):
            action = agent.policy(observation)
            new_observation, reward, terminated, truncated, _ = env.step(
                action)

            agent.collect_observation(observation, action, reward,
                                      new_observation, terminated)
            agent.update()
            agent.update_target_network()
            agent.epsilon_decay()

            observation = new_observation.copy()
            cummulative_reward += reward

        print(f'Finished episode {e} with reward = {cummulative_reward}')

        cummulative_rewards.append(cummulative_reward)
    return cummulative_rewards


def main(network_config, batch_size, gamma, start_epsilon, min_epsilon, decay,
         learning_rate, number_of_actions, target_network_update_frequency,
         number_of_episodes):
    layers = []
    layer_size_tuples = list(zip(network_config, network_config[1:]))
    for i, (in_size, out_size) in enumerate(layer_size_tuples):
        layers.append(torch.nn.Linear(in_size, out_size))
        if i != len(layer_size_tuples) - 1:
            layers.append(torch.nn.ReLU())
    network = torch.nn.Sequential(*layers)

    agent = QLearningAgent(network, batch_size, gamma, start_epsilon,
                           min_epsilon, decay, learning_rate,
                           number_of_actions, target_network_update_frequency)

    rewards = train_agent(number_of_episodes, agent)

    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Qlearning agent for moonlander')
    parser.add_argument('--network_config',
                        type=list,
                        nargs='+',
                        default=[8, 128, 128, 128, 4])
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size for training')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma',
                        type=float,
                        default=.99,
                        help='Future reward discount')
    parser.add_argument('--start_epsilon',
                        type=float,
                        default=.9,
                        help='exploration rate')
    parser.add_argument('--min_epsilon',
                        type=float,
                        default=.05,
                        help='exploration rate')
    parser.add_argument('--decay',
                        type=float,
                        default=.999,
                        help='exploration rate')
    parser.add_argument('--target_network_update_frequency',
                        type=int,
                        default=100,
                        help='Target network update frequency.')
    parser.add_argument('--number_of_episodes',
                        type=int,
                        default=1500,
                        help='number of game episodes the agent plays.')

    args = parser.parse_args()

    main(args.network_config, args.batch_size, args.gamma, args.start_epsilon,
         args.min_epsilon, args.decay, args.learning_rate,
         args.network_config[-1], args.target_network_update_frequency,
         args.number_of_episodes)
