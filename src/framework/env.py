# đổi lại thành dạng hash table
import gym
import numpy as np
from gym import spaces
import random
from scipy import spatial
from framework.utils import build_adj_matrix_and_embeddings

G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()


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


def get_k_nearest_candidate(target_node, cosim_hash_table, k=11):
    k_nearest_nodes = [item[0] for item in cosim_hash_table[target_node][:k]]
    return k_nearest_nodes


def getHashTable(ground_truth, cosim_hash_table):
  '''
    Input: groundtruth, 
  '''
  hash_mapping_node = {}
  for source_node, target_node in ground_truth.items():
    hash_mapping_node[source_node] = get_k_nearest_candidate(target_node, cosim_hash_table)
  return hash_mapping_node


def getAlignTable():
    return ground_truth


def getListState(hash_table):
    lst_state = []
    for key in hash_table.keys():
        lst_key_2 = hash_table[key]
        for key_2 in lst_key_2:
            lst_state.append((int(key), key_2))
    return lst_state


def getNodeEmbeddingByKey(emb, id):
    return emb[id]


def popCurrentState(lst_state, current_state):
    # print('currstate: ', current_state)
    key1, key2 = current_state[0], current_state[1]
    # print((key1, key2))

    lst_remove = []
    for state in lst_state:
        key_state_1 = state[0]
        key_state_2 = state[1]
        # print('cur_state: ', state)

        if key_state_1 == key1 or key_state_2 == key1 or key_state_1 == key2 or key_state_2 == key2:
            lst_remove.append(state)
    lst_final = list(set(lst_state) - set(lst_remove))
    return lst_final


def isAligned(state):
    alignTable = getAlignTable()
    true_state = (state[0], alignTable[state[0]])
    if state == true_state:
        return True
    return False


class SequentialMatchingEnv(gym.Env):
    """
    Custom Environment for Binary Scheme
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for action
    MATCH = 1
    UNMATCH = 0

    MIN_SKIP_RATE = 0.01
    BASIC_SKIP_RATE = 0.8
    DISCOUNT_RATIO = 0.9
    THETA = 1

    def __init__(self, k_nearest=5):
        super(SequentialMatchingEnv, self).__init__()

        self.seed = random.randint(0, 100)

        cosim_hash_table = get_cosim_hash_table(emb1, emb2)
        self.hash_table = getHashTable(ground_truth, cosim_hash_table)
        self.list_state = getListState(self.hash_table)
        # init total reward
        self.total_reward = 0

        # define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # observation will be the source vector and the target vector with each (1,5)vector
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, 5), dtype=np.float16)

    def reset(self):
        """
          Important: The observation must be numpy array 
          : return: (np.array)
        """
        self.lst_state = getListState(
            self.hash_table)  # get all state possible

        first_state = self.lst_state[0]
        self.state_embedding = np.array([getNodeEmbeddingByKey(emb1, first_state[0]),
                                         getNodeEmbeddingByKey(emb2, first_state[1])])
        return self.lst_state[0]

    def step(self, action, episode_num=1):

        self.episode_num = episode_num
        current_state = self.lst_state[0]

        if action == self.MATCH:
            # if match pop all state containing two key of current state
            self.lst_state = popCurrentState(self.lst_state, current_state)

            # check value if true match
            if isAligned(current_state):
                score = 1

            elif not isAligned(current_state):
                score = 0
        elif action == self.UNMATCH:
            self.lst_state.pop(0)
            if not isAligned(current_state):
                score = 0
            elif isAligned(current_state):
                score = -1
        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(action))
        info = {}

        if len(self.lst_state) == 0:
            next_embedding = None
            next_state = (None, None)  # khong con phan tu nao trong G1
            done = True
        else:
            # get next state id and next state embedding
            next_state = self.lst_state[0]
            next_embedding = np.array([getNodeEmbeddingByKey(emb1, next_state[0]),
                                       getNodeEmbeddingByKey(emb2, next_state[1])])
            done = False
        return current_state, next_state, next_embedding, score, done, info
