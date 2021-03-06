import numpy as np

def normalize(emb):
    return emb / np.sqrt((emb ** 2).sum(axis=1)).reshape(-1, 1)


def build_embeddings():
    # Build adj matrix
    file1 = open('./data/rel_triples_id_1.txt', 'r')
    lines1 = file1.readlines()
    lines1 = lines1[:100]
    g1 = []
    for line in lines1:
        tmp_arr =  [int(x) for x in line.strip().split("\t")]
        g1.append([tmp_arr[0], tmp_arr[2]])
    g1 = np.array(g1)

    file2 = open('./data/rel_triples_id_2.txt', 'r')
    lines2 = file2.readlines()
    lines2 = lines2[:100]
    g2 = []
    for line in lines2:
        tmp_arr =  [int(x) for x in line.strip().split("\t")]
        g2.append([tmp_arr[0], tmp_arr[2]])
    g2 = np.array(g2)

    unique1 = np.unique(g1)
    unique2 = np.unique(g2)
    # Build embeddings
    emb1 = []
    emb2 = []
    proximi = np.load("./data/proximi_emb.npy")
    transi = np.load("./data/transitivity_emb.npy")
    final_emb = np.concatenate((normalize(proximi), transi * 0.65), axis=1)

    for i in np.array(unique1):
        emb1.append(final_emb[i])
    for i in np.array(unique2):
        emb2.append(final_emb[i])
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    return emb1, emb2, unique1, unique2

def build_adj_matrix_and_embeddings():
    # Build adj matrix
    file1 = open('./data/rel_triples_id_1.txt', 'r')
    lines1 = file1.readlines()
    lines1 = lines1[:100]
    g1 = []
    for line in lines1:
        g1.append([int(x) for x in line.strip().split("\t")])
    g1 = np.array(g1)

    file2 = open('./data/rel_triples_id_2.txt', 'r')
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

    G1_adj_matrix = np.zeros(shape=(G1_nodes, G1_nodes))
    G2_adj_matrix = np.zeros(shape=(G2_nodes, G2_nodes))

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

    # Build embeddings
    emb1 = []
    emb2 = []
    proximi = np.load("./data/proximi_emb.npy")
    for i, _ in enumerate(mapping_index_1):
        emb1.append(proximi[i])
    for i, _ in enumerate(mapping_index_2):
        emb2.append(proximi[i])
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

    return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth

if __name__ =="__main__":
    build_embeddings()