import numpy as np


def build_adj_matrix_and_embeddings():
    # Build adj matrix
    file1 = open('./data/data/rel_triples_id_1.txt', 'r')
    lines1 = file1.readlines()
    lines1 = lines1[:100]
    g1 = []
    for line in lines1:
        g1.append([int(x) for x in line.strip().split("\t")])
    g1 = np.array(g1)

    file2 = open('./data/data/rel_triples_id_2.txt', 'r')
    lines2 = file2.readlines()
    lines2 = lines2[:100]
    g2 = []
    for line in lines2:
        g2.append([int(x) for x in line.strip().split("\t")])
    g2 = np.array(g2)
    mapping_index_1 = {}
    mapping_index_2 = {}
    unique1 = np.unique(g1)
    unique2 = np.unique(g2)
    for i in range(len(unique1)):
        mapping_index_1[unique1[i]] = i

    for i in range(len(unique2)):
        mapping_index_2[unique2[i]] = i

    G1_nodes = len(np.unique(g1))
    G2_nodes = len(np.unique(g2))
    # print("num nodes: ", G1_nodes)
    # print("num nodes: ", G2_nodes)
    G1_adj_matrix = np.empty(shape=(G1_nodes, G1_nodes))
    G2_adj_matrix = np.empty(shape=(G2_nodes, G2_nodes))
    # print(mapping_index_1)
    # print(mapping_index_2)
    for i in range(len(g1)):
        head = g1[i][0]
        tail = g1[i][2]
        G1_adj_matrix[mapping_index_1[head]][mapping_index_1[tail]] = 1
        G1_adj_matrix[mapping_index_1[tail]][mapping_index_1[head]] = 1
    for i in range(len(g1)):
        head = g2[i][0]
        tail = g2[i][2]
        G2_adj_matrix[mapping_index_2[head]][mapping_index_2[tail]] = 1
        G2_adj_matrix[mapping_index_2[tail]][mapping_index_2[head]] = 1
    # print(np.count_nonzero(G1_adj_matrix == 1))
    # print(np.count_nonzero(G2_adj_matrix == 1))

    # Build embeddings
    emb1 = []
    emb2 = []
    # transitivity = np.load("/content/drive/MyDrive/Colab Notebooks/RLWithGraphEntityAlignment/data/transitivity_emb.npy")
    proximi = np.load("./data/data/proximi_emb.npy")
    for i, _ in enumerate(mapping_index_1):
        emb1.append(proximi[i])
    for i, _ in enumerate(mapping_index_2):
        emb2.append(proximi[i])
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    # Get ground truth
    ground_truth = {}
    file_gt = open('./data/data/ground_truth.txt', 'r')
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

    return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth
