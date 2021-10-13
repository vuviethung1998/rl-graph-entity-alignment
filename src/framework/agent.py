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

    def __init__(self, first_adj_matrix, second_adj_matrix, inputs_shape, outputs_shape):
        super(GCN_layer, self).__init__()

        self.W = Parameter(torch.rand(
            inputs_shape, outputs_shape), requires_grad=True)
        self.bias = Parameter(torch.rand(outputs_shape), requires_grad=True)
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
        propagate = torch.mm(aggregate, self.W)+self.bias
        return propagate


class Agent(nn.Module):
    def __init__(self, first_adj_matrix, second_adj_matrix, inputs_shape, outputs_shape, n_classes, gamma, activation='Sigmoid'):
        super(Agent, self).__init__()

        self.layer1 = GCN_layer(
            first_adj_matrix, second_adj_matrix, inputs_shape, outputs_shape)
        self.layer2 = GCN_layer(
            first_adj_matrix, second_adj_matrix, outputs_shape, n_classes)
        self.n_classes = n_classes
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
        self.W_h = Parameter(torch.rand(1, 2), requires_grad=True)
        self.W_f = Parameter(torch.rand(
            n_classes, n_classes), requires_grad=True)
        self.W_p = Parameter(torch.rand(1, 1), requires_grad=True)
        self.bias_h = Parameter(torch.rand(1), requires_grad=True)

    @classmethod
    # Do later
    def get_cosim_hash_table(emb1, emb2):
        cosim_hash_table = {}
        for i in range(len(emb1)):
            cur_vec_1 = emb1[i]
            lst_cosim = [(j, 1 - spatial.distance.cosine(cur_vec_1, emb2[j]))
                        for j in range(len(emb2))]
            lst_sorted_cosim = sorted(lst_cosim, key=(
                lambda node: node[1]), reverse=True)
            cosim_hash_table.update({i: lst_sorted_cosim})
        return cosim_hash_table

    @classmethod
    # Do later
    def get_k_nearest_candidate(target_node, cosim_hash_table, k=11):
        k_nearest_nodes = [item[0] for item in cosim_hash_table[target_node][:k]]
        return k_nearest_nodes
        
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
        g_x = torch.reshape(G_x[index_x], (1, self.n_classes))
        g_y = torch.reshape(G_y[index_y], (1, self.n_classes))

        # Linear combination
        cat_gxgy = torch.cat((g_x, g_y), 0)
        h = self.sigmoid(torch.mm(self.W_h, cat_gxgy) + self.bias_h)

        # Mutual information estimator
        f = torch.exp(g_x.T*self.W_f*g_y)
        # Bao gồm cả chính node e_y nên lấy k=11. Paper nói là k=10
        k_nearest_opponent_vector = self.get_k_nearest_opponent(G_y, g_y, k=11)
        list_temp = [torch.exp(g_x.T*self.W_f*oppo)
                     for oppo in k_nearest_opponent_vector]
        f_oppo = torch.stack(list_temp).sum()
        I = f/f_oppo

        # Policy
        policy = self.softmax(torch.mm(self.W_p, torch.cat((h, I), 1)))
        return policy

    # Train model using reinforcement learning
    def train_model(self, first_embeddings, second_embeddings, net, transitions, optimizer):
        
        states, actions, rewards = transitions.state, transitions.action, transitions.reward
        actions = torch.stack(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        policies = torch.zeros_like(actions).to(device)
        vf = torch.zeros_like(rewards).to(device)
        total_loss = 0
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            # calculate G, begin from the last transition
            running_return = self.gamma**0 * \
                rewards[t] + self.gamma * running_return
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
        total_loss += loss

        return total_loss

    def get_action(self, first_embeddings, second_embeddings, state):
        policy = self.forward(first_embeddings, second_embeddings, state)
        p = policy[0].cpu().data.numpy()
        action = np.random.choice(2, 1, p=p)[0]
        return action