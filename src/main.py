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

torch.manual_seed(500)

if __name__ == '__main__':
    print("Loading data...")
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()
    number_gt = len(ground_truth)
    print("Number of matched pairs: ", number_gt)
    np.savetxt("./output/debug/G1_adj_matrix.csv", G1_adj_matrix, delimiter=',')
    np.savetxt("./output/debug/G2_adj_matrix.csv", G2_adj_matrix, delimiter=',')
    np.savetxt("./output/debug/emb1.csv", emb1, delimiter=',')
    np.savetxt("./output/debug/emb2.csv", emb2, delimiter=',')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_embeddings_torch = torch.from_numpy(
        emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        emb2).type(torch.FloatTensor).to(device)

    print("Intitializing agent...")
    lr = 0.0001
    net = Agent(G1_adj_matrix, G2_adj_matrix,
                emb1.shape[1], 128, 64, gamma=0.99, activation='Sigmoid')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    net.train()

    print("Building environment...")
    env = SequentialMatchingEnv()
    rewards_lst = []
    episodes = 500
    lst_state = []
    results = []
    prob = {}
    for k, v in ground_truth.items():
        prob[(k, v)] = []
    
    print("Training...")
    for ep in tqdm(range(1, episodes + 1)):
        idx = env.reset()
        memory = Memory()
        reward_episode = 0
        done = False
        # define policy and action
        while True:
            # get environment state
            action, p = net.get_action(first_embeddings_torch,
                                    second_embeddings_torch, idx)
            if isAligned(idx):
                prob[idx].append(p[1])
            cur_idx, next_idx, obs, reward, done, info = env.step(action, ep)
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

        loss = net.train_model(
            first_embeddings_torch, second_embeddings_torch, net, memory.sample(), optimizer)
        results.append([ep, reward_episode, loss.cpu().detach().numpy()])
        if ep % 50 == 0:
            print("Episode: {}   Reward: {}   Agent loss: {}".format(str(ep), str(reward_episode), str(loss.cpu().detach().numpy())))
        if reward_episode == number_gt:
            print("Goal reached!", "reward=", reward_episode)
            break
    
    print("Saving results and plotting...")
    # Save results
    results = np.array(results)
    results = pd.DataFrame(results, columns=['episode', 'reward', 'agent_loss'])
    results.to_csv("./output/results/episodes.csv", index=False)
    prob_df = pd.DataFrame.from_dict(prob)
    prob_df.to_csv("./output/results/prob.csv", index=False)

    # Visualization
    sns.lineplot(data=results.reward, color="g")
    plt.legend(labels=["Reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig('./output/results/rewards.png')
