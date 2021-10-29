import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from framework.utils import build_adj_matrix_and_embeddings, normalize_prob
from framework.env import SequentialMatchingEnv
from framework.agent import Agent
from framework.memory import Memory
from framework.env import isAligned
import time
import sys
import argparse
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def seed():
    np.random.seed(2)
    torch.manual_seed(500)


def plot(results, log_results, prob, num_gt):
    results = np.array(results)
    results = pd.DataFrame(
        results, columns=['episode', 'reward', 'agent_loss', 'time'])
    results.to_csv(log_results + "/episodes.csv", index=False)
    prob_df = pd.DataFrame.from_dict(prob)
    prob_df.to_csv(log_results + "/prob.csv", index=False)

    # Visualization
    sns.lineplot(data=results.reward/num_gt, color="g")
    plt.legend(labels=["Reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(log_results + '/rewards.png')


def train(args):
    print("Loading data...")
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings(
        True)
    num_gt = len(ground_truth)
    print("Num nodes in G1: ", len(G1_adj_matrix))
    print("Num nodes in G2: ", len(G2_adj_matrix))
    print("Num ground_truth: ", num_gt)

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
    early_stopping = 0
    results = {}
    results["num_gt"] = num_gt
    results["training"] = []  # results of each episode
    results["prob"] = {}  # to check probability of each pair after one episode
    for k, v in ground_truth.items():
        results["prob"][(k, v)] = [0]

    print("Training...")
    for ep in tqdm(range(1, args.episode + 1)):
        start_episode = time.time()

        # Reset environment
        idx = env.reset(ep)
        memory = Memory()
        reward_episode = 0
        done = False

        # Define is_start for pre-calculate consine similarity in function "forward" of agent
        is_start = True
        action, p = agent.get_action(first_embeddings_torch,
                                     second_embeddings_torch, [idx], is_start)
        is_start = False

        # Get policy, action and reward
        while True:
            if isAligned(idx):
                results["prob"][idx].append(p[1])
            next_idx, reward, done = env.step(action)
            # add reward
            reward_episode += reward

            if done:
                break

            # push to memory for training model
            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            memory.push(next_idx, action_one_hot, reward)

            # next state
            idx = next_idx

            # get environment state
            action, p = agent.get_action(first_embeddings_torch,
                                         second_embeddings_torch, [idx], is_start)

            # just for storing "prob" results
        results["prob"] = normalize_prob(results["prob"])
        # Train model
        loss = agent.train_model_seq(
            first_embeddings_torch, second_embeddings_torch, agent, memory.sample(), optimizer)
        end_episode = time.time()
        results["training"].append([ep, reward_episode, loss.cpu(
        ).detach().numpy(), end_episode - start_episode])

        # Monitoring
        if ep % 5 == 0:
            print("Episode: {}   Reward: {}/{}   Agent loss: {}".format(ep,
                  reward_episode, num_gt, loss.cpu().detach().numpy()))

        # Early stopping
        if reward_episode == num_gt:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping == args.early_stopping:
            torch.save(agent.state_dict(), args.log_weights + "/best.pt")
            print("Early stopping")
            print("Goal reached! Reward: {}/{}".format(reward_episode, num_gt))
            break

    return results, agent

if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_results',
                        default="./log/train/results/_test",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--log_weights',
                        default="./log/train/weights/_test",
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
    parser.add_argument('--early_stopping',
                        default=50,
                        type=int,
                        help='Early stopping')
    parser.add_argument('--episode',
                        default=500,
                        type=int,
                        help='Early stopping')

    args = parser.parse_args()
    if not os.path.exists(args.log_results):
        os.makedirs(args.log_results)
    if not os.path.exists(args.log_weights):
        os.makedirs(args.log_weights)
    print("Beginning the training process...")
    results, agent = train(args)

    print("Saving results...")
    torch.save(agent.state_dict(), args.log_weights + "/best.pt")
    plot(results["training"], args.log_results,
         results["prob"], results["num_gt"])
    print("Done!")
