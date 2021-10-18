import numpy as np
import pandas as pd
import networkx as nx

def build_adj_matrix_and_embeddings():
    # Build adj matrix
    file1 = open('./data/rel_triples_id_1.txt', 'r')
    lines1 = file1.readlines()
    g1 = []
    for line in lines1:
        g1.append([int(x) for x in line.strip().split("\t")])
    edges1_df = pd.DataFrame(g1, columns=["source", "relation", "target"])
    graph1 = nx.convert_matrix.from_pandas_edgelist(edges1_df, "source", "target")
    subgraph_nodes1 = list(nx.dfs_preorder_nodes(graph1))[:100] # Gets 150 nodes in the graph 
    G1 = graph1.subgraph(subgraph_nodes1)


    file2 = open('./data/rel_triples_id_2.txt', 'r')
    lines2 = file2.readlines()
    g2 = []
    for line in lines2:
        g2.append([int(x) for x in line.strip().split("\t")])
    edges2_df = pd.DataFrame(g2, columns=["source", "relation", "target"])
    graph2 = nx.convert_matrix.from_pandas_edgelist(edges2_df, "source", "target")
    subgraph_nodes2 = list(nx.dfs_preorder_nodes(graph2))[:100] # Gets 150 nodes in the graph 
    G2 = graph2.subgraph(subgraph_nodes2)

    G1_adj_matrix = nx.to_numpy_array(G1, nodelist=sorted(G1.nodes))
    G2_adj_matrix = nx.to_numpy_array(G2, nodelist=sorted(G2.nodes))

    # Get embeddings
    mapping_index_1 = {} # {node: index}
    mapping_index_2 = {}
    nodes1 = np.sort(G1.nodes)
    nodes2 = np.sort(G2.nodes)
    for i in range(len(nodes1)):
        mapping_index_1[nodes1[i]] = i

    for i in range(len(nodes2)):
        mapping_index_2[nodes2[i]] = i
    emb1 = []
    emb2 = []
    proximi = np.load("./data/proximi_emb.npy")
    for k1, k2 in zip(mapping_index_1.keys(), mapping_index_2.keys()):
        emb1.append(proximi[k1])
        emb2.append(proximi[k2])
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    # Get ground truth
    ground_truth = {}
    file_gt = open('./data/ground_truth.txt', 'r')
    lines = file_gt.readlines()
    gt = []
    for line in lines:
        gt.append([int(x) for x in line.strip().split("\t")])
    gt = np.array(gt)

    for i in range(len(gt)):
        index_x = gt[i][0]
        index_y = gt[i][1]
        if index_x in mapping_index_1 and index_y in mapping_index_2:
            ground_truth[mapping_index_1[index_x]] = mapping_index_2[index_y]

    print(ground_truth)
    return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth

if __name__ == '__main__':
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()