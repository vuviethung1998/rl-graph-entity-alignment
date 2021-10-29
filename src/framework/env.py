# đổi lại thành dạng hash table
import gym
import numpy as np
from gym import spaces
import random
from scipy import spatial
from framework.utils import build_adj_matrix_and_embeddings
from sklearn import preprocessing

G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings(True)

def get_cosim_hash_table(emb1, emb2, top_k=11):
    # get hashtable of nearest node corresponding to each source node 
    emb1 = preprocessing.normalize(emb1)
    emb2 = preprocessing.normalize(emb2)
    sim_mat = np.matmul(emb1, emb2.T) 
    num  = sim_mat.shape[0]
    
    idx = list(range(num)) # list true index 

    cosim_hash_table = {}
    prob_skip_hash_table = {} 

    for i in idx:
        g1_target = idx[i]
        rank = np.argpartition(-sim_mat[i, :], np.array(top_k)-1)
        cosim_hash_table.update({g1_target: [ j for j in rank[0:top_k] ]})

        # get prob of skipping each state = C(ex, emax) - C(ex, ey)
        max_sim_score = max(sim_mat[i,rank[0:top_k]] ) 
        lst_sim_mat = [ max_sim_score -i for i in sim_mat[i,rank[0:top_k]] ]
        prob_skip_hash_table.update({g1_target: lst_sim_mat})

    return cosim_hash_table, prob_skip_hash_table      

def get_k_nearest_candidate(target_node, cosim_hash_table, prob_skip_hash_table):
    # k_nearest_nodes=  [item[0] for item in  ]
    return cosim_hash_table[target_node], prob_skip_hash_table[target_node]

def getHashTable(ground_truth, cosim_hash_table, prob_skip_hash_table):
    '''
        Input: groundtruth,
        Output: hashtable of corresponding  and hashtable of skipping rate 
    '''
    hash_mapping_node = {}
    hash_skip_rate = {}

    for source_node, target_node in ground_truth.items():
        hash_mapping_node[source_node], hash_skip_rate[source_node]  = get_k_nearest_candidate(
            target_node, cosim_hash_table, prob_skip_hash_table)
    return hash_mapping_node, hash_skip_rate

def getAlignTable():
    return ground_truth


def getListState(hash_table, hash_skip_rate, ep_num=1, min_skip_rate=0.05, basic_skip_rate=0.9, discount_ratio=0.9):
    # calculate skip rate at each episode and eliminate 
    lst_state = []
    for key in hash_table.keys():
        lst_key_2 = hash_table[key]
        lst_prob = hash_skip_rate[key]
        
        # prob to skip all state of a node in x 
        # skip rate = max (min, 0.9 ^ (n-1) * 0.9 * (1- max_prob))

        #idea 1
        # max_prob = max(lst_prob) 
        # prob_skip_all = max(min_skip_rate, (1 - discount_ratio ** (ep_num -1) ) * basic_skip_rate * (1-max_prob))

        # if np.random.rand() < prob_skip_all: # skip this state x
        #     continue

        for i in range(len(lst_key_2)):
            # idea 2: skip rate
            prob = max(min_skip_rate, (1 - discount_ratio ** (ep_num -1)) * basic_skip_rate * lst_prob[i] ) #  max(pmin s, η^(t−1)ps * di)            
            cur_rand = np.random.rand() 
            # print('prob - cur_rand: {} {}'.format(prob, cur_rand))
            if(prob< cur_rand):
                lst_state.append((key, lst_key_2[i]))
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

        if key_state_1 == key1 or key_state_2 == key2:
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
    # Define constants for action
    MATCH = 1
    UNMATCH = 0

    def __init__(self, k_nearest=5):
        super(SequentialMatchingEnv, self).__init__()

        self.seed = random.randint(0, 100)

        cosim_hash_table, prob_skip_hash_table = get_cosim_hash_table(emb1, emb2)
        self.hash_table, self.hash_skip_rate = getHashTable(ground_truth, cosim_hash_table, prob_skip_hash_table)
        self.list_state = getListState(self.hash_table, self.hash_skip_rate)
        # init total reward
        self.total_reward = 0

        # define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # observation will be the source vector and the target vector with each (1,5)vector
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, 5), dtype=np.float16)

    def reset(self, ep_num=1):
        """
          Important: The observation must be numpy array 
          : return: (np.array)
        """
        self.lst_state = getListState(
            self.hash_table, self.hash_skip_rate,ep_num=ep_num)  # get all state possible
        # print('lst_state: {}'.format(self.lst_state))
        print("len lst_state: {}".format(len(self.lst_state)))
        first_state = self.lst_state[0]
        self.state_embedding = np.array([getNodeEmbeddingByKey(emb1, first_state[0]),
                                         getNodeEmbeddingByKey(emb2, first_state[1])])

        return self.lst_state[0]

    def step(self, action):

        current_state = self.lst_state[0]

        if action == self.MATCH:            
            # check value if true match
            if isAligned(current_state):  # label = 1 => difficulty = Cemax,ex - Cexey
                self.lst_state = popCurrentState(self.lst_state, current_state)
                score = 1

            # label = 0 => difficulty = theta - (Cemax,ex - Cexey)
            elif not isAligned(current_state):
                self.lst_state.pop(0)
                score = 0
                # else: score = 0 # neu skip, score = 0 va
        elif action == self.UNMATCH:
            # tinh xac suat skip 


            self.lst_state.pop(0)
            if not isAligned(current_state):
                score = 0
            elif isAligned(current_state):
                score = -1
        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(action))

        if len(self.lst_state) == 0:
            next_state = (None, None)  # khong con phan tu nao trong G1
            done = True
        else:
            # get next state id and next state embedding
            next_state = self.lst_state[0]
            done = False
        return next_state, score, done


if __name__ =="__main__":
    env = SequentialMatchingEnv()
    for i in range(100):
        print(i)
        env.reset(i)