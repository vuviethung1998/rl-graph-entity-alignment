import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from framework.utils import build_adj_matrix_and_embeddings
from framework.env import SequentialMatchingEnv
from framework.agent import Agent
from framework.memory import Memory
torch.manual_seed(500)


if __name__ == '__main__':
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_embeddings_torch = torch.from_numpy(
        emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        emb2).type(torch.FloatTensor).to(device)

    lr = 0.0001
    net = Agent(G1_adj_matrix, G2_adj_matrix,
                emb1.shape[1], 16, 1, gamma=0.99, activation='Sigmoid')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    net.train()

    env = SequentialMatchingEnv()
    rewards_lst = []
    episodes = 1
    lst_state = []
    results = []

    for ep in tqdm(range(1, episodes + 1)):
        idx = env.reset()
        memory = Memory()
        reward_episode = 0
        done = False
        # define policy and action
        while True:
            # get environment state
            action = net.get_action(first_embeddings_torch,
                                    second_embeddings_torch, idx)
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
        if reward_episode > 15:
            print("Goal reached!", "reward=", reward)
            break

    results = np.array(results)
