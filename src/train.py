import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from framework.utils import build_adj_matrix_and_embeddings
from framework.env import SequentialMatchingEnv
from framework.agent import Agent
from framework.memory import Memory
from framework.env import isAligned
import time
import sys
import argparse
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed():
    np.random.seed(2)
    torch.manual_seed(500)


def plot(results, log_results, prob):
    results = np.array(results)
    results = pd.DataFrame(
        results, columns=['episode', 'reward', 'agent_loss', 'time'])
    results.to_csv(log_results + "/episodes.csv", index=False)
    prob_df = pd.DataFrame.from_dict(prob)
    prob_df.to_csv(log_results + "/prob.csv", index=False)

    # Visualization
    sns.lineplot(data=results.reward, color="g")
    plt.legend(labels=["Reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(log_results + '/rewards.png')


def train():
    print("Loading data...")
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()
    number_gt = len(ground_truth)

    first_embeddings_torch = torch.from_numpy(
        emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        emb2).type(torch.FloatTensor).to(device)

    print("Intitializing agent...")
    agent = Agent(G1_adj_matrix, G2_adj_matrix,
                  emb1.shape[1], 128, 64, gamma=args.gamma, activation='Sigmoid')
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    agent.to(device)
    agent.train()

    print("Building environment...")
    env = SequentialMatchingEnv()
    episodes = 5
    results = {}
    results["training"] = [] # results of each episode
    results["prob"] = {} # to check probability of each pair after one episode
    for k, v in ground_truth.items():
        results["prob"][(k, v)] = []

    print("Training...")
    for ep in tqdm(range(1, episodes + 1)):
        start_episode = time.time()
        
        # Reset environment
        idx = env.reset()
        lst_state = []
        memory = Memory()
        reward_episode = 0
        done = False
        # Define start_episode for pre-calculate consine similarity in function "foward" of agent
        start_episode = True
        action, p = agent.get_action(first_embeddings_torch,
                                     second_embeddings_torch, [idx], start_episode)
        start_episode = False

        # Get policy, action and reward
        while True:
            if isAligned(idx):
                results["prob"][idx].append(p[1])
            cur_idx, next_idx, _, reward, done, _ = env.step(action, ep)
            if done:
                break

            # add reward
            reward_episode += reward

            # push to memory for training model
            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            memory.push(next_idx, action_one_hot, reward)

            # next state
            idx = next_idx
            lst_state.append((cur_idx, action, reward))

            # get environment state
            action, p = agent.get_action(first_embeddings_torch,
                                         second_embeddings_torch, [idx], start_episode)

        # Train model
        loss = agent.train_model_seq(
            first_embeddings_torch, second_embeddings_torch, agent, memory.sample(), optimizer)
        end_episode = time.time()
        results["training"].append([ep, reward_episode, loss.cpu(
        ).detach().numpy(), end_episode - start_episode])

        # Monitoring
        if ep % 50 == 0:
            print("Episode: {}   Reward: {}/{}   Agent loss: {}".format(
                str(ep), str(reward_episode), str(number_gt), str(loss.cpu().detach().numpy())))
        if reward_episode == number_gt:
            print(
                "Goal reached! Reward={}/{}".format(str(reward_episode), str(number_gt)))
            torch.save(agent.state_dict(), args.log_weights + "/best.pt")
            break
    return results, agent


if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_results',
                        default="./log/train/results",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--log_weights',
                        default="./log/train/weights",
                        type=str,
                        help='Directory for weights')
    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--gamma',
                        default=0.99,
                        type=float,
                        help='Gamma in Reinforcement Learning')

    args = parser.parse_args()

    print("Beginning the training process...")
    results, agent = train()

    print("Saving results and plotting...")
    torch.save(agent.state_dict(), args.log_weights + "/best.pt")
    plot(results["training"], args.log_results, results["prob"])
