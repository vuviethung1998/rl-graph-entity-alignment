from scipy import spatial
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ValueFunctionNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueFunctionNet, self).__init__()
        self.dense_layer = nn.Linear(state_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = nn.Sigmoid(self.dense_layer(x))
        return self.output(x)


class GCN_layer(nn.Module):
    """
      Define filter layer 1/2 like in the above image
      Calculate A_hat first then,
      Input: adj_matrix with input features X
    """

    def __init__(self, first_adj_matrix, second_adj_matrix, input_shape, hidden_states):
        super(GCN_layer, self).__init__()
        self.fc = nn.Linear(input_shape, hidden_states)
        A = torch.from_numpy(first_adj_matrix).type(
            torch.LongTensor).to(device)
        I = torch.eye(A.shape[0]).to(device)
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_x = torch.mm(torch.mm(D_inv, A_hat), D_inv)

        A = torch.from_numpy(second_adj_matrix).type(
            torch.LongTensor).to(device).to(device)
        I = torch.eye(A.shape[0]).to(device)
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_y = torch.mm(torch.mm(D_inv, A_hat), D_inv)

    def forward(self, i, input_features):
        if i == "x":
            aggregate = torch.mm(self.A_hat_x, input_features)
        else:
            aggregate = torch.mm(self.A_hat_y, input_features)
        propagate = self.fc(aggregate)
        return propagate


class Agent(nn.Module):
    def __init__(self, first_adj_matrix, second_adj_matrix, input_shape, hidden_states, output_shape, gamma, activation='Sigmoid'):
        super(Agent, self).__init__()

        self.layer1 = GCN_layer(
            first_adj_matrix, second_adj_matrix, input_shape, hidden_states)
        self.layer2 = GCN_layer(
            first_adj_matrix, second_adj_matrix, hidden_states, output_shape)
        self.output_shape = output_shape
        self.gamma = gamma

        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.fc_h = nn.Linear(output_shape*2, 128)
        # self.fc_p = nn.Linear(128+output_shape, 2) # enable MI
        self.fc_p = nn.Linear(128, 2) # unenable MI
        self.W_f = Parameter(torch.rand(
            output_shape, 1).to(device), requires_grad=True)

    # Gett k nearest opponents for mutual information estimator
    def get_k_nearest_opponent(self, G, node, k=3):
        G_list = [(i, item) for i, item in enumerate(G)]
        # Tìm nearest thế này là gồm cả chính node đó
        nearest_node = sorted(G_list, key=(lambda other_node: F.cosine_similarity(
            torch.reshape(other_node[1], (1, -1)), node)), reverse=True)
        k_nearest_opponent = nearest_node[:k]
        k_nearest_opponent_vector = [G[item[0]] for item in k_nearest_opponent]
        return k_nearest_opponent_vector

    def forward(self, first_embeddings, second_embeddings, state):
        index_x = state[0]
        index_y = state[1]
        x = self.layer1("x", first_embeddings)
        x = self.activation(x)
        x = self.layer2("x", x)
        G_x = self.activation(x)

        y = self.layer1("y", second_embeddings)
        y = self.activation(y)
        y = self.layer2("y", y)
        G_y = self.activation(y)
        g_x = torch.reshape(G_x[index_x], (1, self.output_shape))
        g_y = torch.reshape(G_y[index_y], (1, self.output_shape))

        # Linear combination
        cat_gxgy = torch.cat((g_x, g_y), 1)
        h = self.sigmoid(self.fc_h(cat_gxgy))

        # # Mutual information estimator
        # f = torch.matmul(torch.matmul(g_x, self.W_f), g_y)
        # # Paper nói là k=10 nhưng bao gồm cả chính node e_y nên lấy k=11.
        # k_nearest_opponent_vector = self.get_k_nearest_opponent(G_y, g_x, k=11)
        # f_oppo = torch.zeros_like(f)
        # for oppo in k_nearest_opponent_vector:
        #     oppo = torch.reshape(oppo, (1, oppo.shape[0]))
        #     f_oppo = torch.add(f_oppo, torch.exp(torch.matmul(torch.matmul(g_x, self.W_f), oppo)))
        # I = f/f_oppo

        # Policy
        # policy = self.softmax(self.fc_p(torch.cat((h, I), 1))) # enable MI
        policy = self.softmax(self.fc_p(h)) # unenable MI
        return policy

    # Train model using reinforcement learning
    def train_model(self, first_embeddings, second_embeddings, net, transitions, optimizer):

        states, actions, rewards = transitions.state, transitions.action, transitions.reward
        actions = torch.stack(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        policies = torch.zeros_like(actions).to(device)
        vf = torch.zeros_like(rewards).to(device)
        running_return = 0

        for t in reversed(range(len(rewards))):
            # calculate G, begin from the last transition
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            if returns.sum() == 0:
                vf[t] = 0.01
            else:
                vf[t] = running_return/returns.sum()

            policies[t] = net(first_embeddings, second_embeddings, states[t])

        # get value function estimates
        # advantage = returns - vf
        advantage = returns
        # loss
        # sum all features/embedding vectors of the state
        log_policies = (torch.log(policies) *
                        actions.detach()).sum(dim=1)
        loss = (-log_policies * advantage).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, first_embeddings, second_embeddings, state):
        policy = self.forward(first_embeddings, second_embeddings, state)
        p = policy[0].cpu().data.numpy()
        action = np.random.choice(2, 1, p=p)[0]
        return action, p