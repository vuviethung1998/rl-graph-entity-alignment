import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

def build_adj_matrix_and_embeddings():
    # Build adj matrix
    file1 = open('./data/rel_triples_id_1.txt', 'r')
    lines1 = file1.readlines()
    g1 = []
    for line in lines1:
        g1.append([int(x) for x in line.strip().split("\t")])
    g1 = pd.DataFrame(g1, columns=["source", "relation", "target"])
    g1 = nx.convert_matrix.from_pandas_edgelist(g1, "source", "target")
    subgraph_nodes1 = list(nx.dfs_preorder_nodes(g1))[:150] # Gets 150 nodes in the graph 
    sub_g1 = g1.subgraph(subgraph_nodes1)
    G1_adj_matrix = nx.to_numpy_array(sub_g1)
    print(sub_g1)

    file2 = open('./data/rel_triples_id_2.txt', 'r')
    lines2 = file2.readlines()
    g2 = []
    for line in lines2:
        g2.append([int(x) for x in line.strip().split("\t")])
    g2 = pd.DataFrame(g2, columns=["source", "relation", "target"])
    g2 = nx.convert_matrix.from_pandas_edgelist(g2, "source", "target")
    print(g2)

    # Build embeddings
    emb1 = []
    emb2 = []
    # transitivity = np.load("./data/transitivity_emb.npy")
    proximi = np.load("./data/proximi_emb.npy")

    # Get ground truth
    ground_truth = {}
    file_gt = open('./data/ground_truth.txt', 'r')
    lines = file_gt.readlines()
    gt = []
    for line in lines:
        gt.append([int(x) for x in line.strip().split("\t")])
    gt = np.array(gt)
    for item in gt:
        ground_truth[item[0]] = item[1]
    # return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth


if __name__ == '__main__':
   print(np.random.choice(2, 1, p=[0.9, 0.1]))