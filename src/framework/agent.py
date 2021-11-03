from operator import index
from scipy import spatial
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')
device2 = torch.device('cpu')

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
            torch.LongTensor).to(device1)
        I = torch.eye(A.shape[0]).to(device1)
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_x = torch.mm(torch.mm(D_inv, A_hat), D_inv).cuda(device)

        A = torch.from_numpy(second_adj_matrix).type(
            torch.LongTensor).to(device2).to(device2)
        I = torch.eye(A.shape[0]).to(device2)
        A_hat = A+I
        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)
        self.A_hat_y = torch.mm(torch.mm(D_inv, A_hat), D_inv).cuda(device)
        torch.cuda.empty_cache()
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
        self.gamma = gamma
        self.similarity_matrix = None

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


    def build_similarity_matrix(self, G_x, G_y):
        return torch.matmul(G_x, G_y.T)
    
    def get_k_nearest_opponent(self, similarity_matrix, index_node, k=11):
        return torch.topk(similarity_matrix[index_node], k).indices

    def forward(self, first_embeddings, second_embeddings, states, start_episode):
        x = self.layer1("x", first_embeddings)
        x = self.activation(x)
        x = self.layer2("x", x)
        G_x = self.activation(x)

        y = self.layer1("y", second_embeddings)
        y = self.activation(y)
        y = self.layer2("y", y)
        G_y = self.activation(y)
        lst_state_x = [G_x[s[0]] for s in states]
        lst_state_y = [G_y[s[1]] for s in states]
        # print(states)
        g_x = torch.stack(lst_state_x)
        g_y = torch.stack(lst_state_y)

        # Linear combination
        cat_gxgy = torch.cat((g_x, g_y), 1)
        h = self.sigmoid(self.fc_h(cat_gxgy))
        
        # # Mutual information estimator
        # f = torch.matmul(torch.matmul(g_x, self.W_f), g_y)
        # f_oppo = torch.zeros_like(f)
        # if start_episode:
        #     self.similarity_matrix = self.build_similarity_matrix(G_x, G_y)
        #     O_y_indices = self.get_k_nearest_opponent(self.similarity_matrix, index_x, k=11)
        # else:
        #     O_y_indices = self.get_k_nearest_opponent(self.similarity_matrix, index_x, k=11)
        # O_y = G_y[O_y_indices]
        # for oppo in O_y:
        #     oppo = torch.reshape(oppo, (1, oppo.shape[0]))
        #     f_oppo = torch.add(f_oppo, torch.exp(torch.matmul(torch.matmul(g_x, self.W_f), oppo)))
        # I = f/f_oppo

        # Policy
        # policy = self.softmax(self.fc_p(torch.cat((h, I), 1))) # enable MI
        policy = self.softmax(self.fc_p(h)) # unenable MI
        return policy

    # Train model using reinforcement learning
    def train_model_seq(self, first_embeddings, second_embeddings, net, transitions, optimizer):
        # with torch.autograd.set_detect_anomaly(True):
        states, actions, rewards = transitions.state, transitions.action, transitions.reward
        actions = torch.stack(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
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
            
            policy = net(first_embeddings, second_embeddings, [states[t]], start_episode=False)
            # get value function estimates
            # advantage = returns - vf
            advantage = returns[t]
            
            # loss
            a = torch.unsqueeze(actions[t], 0)
            log_policies = (torch.log(policy) *a.detach()).sum(dim=1)
            loss = (-log_policies * advantage).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def train_model_batch(self, first_embeddings, second_embeddings, net, transitions, optimizer):
        # with torch.autograd.set_detect_anomaly(True):
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
            
        policies = net(first_embeddings, second_embeddings, states, start_episode=False)
        # get value function estimates
        # advantage = returns - vf
        advantage = returns[t]
        
        # loss
        log_policies = (torch.log(policies) *actions.detach()).sum(dim=1)
        loss = (-log_policies * advantage).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, first_embeddings, second_embeddings, state, start_episode):
        policy = self.forward(first_embeddings, second_embeddings, state, start_episode)
        p = policy[0].cpu().data.numpy()
        action = np.random.choice(2, 1, p=p)[0]
        return action, p