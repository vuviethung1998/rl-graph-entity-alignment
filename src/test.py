import torch
from tqdm import tqdm
from framework.utils import build_adj_matrix_and_embeddings
from framework.agent import Agent
from framework.utils import save_results

if __name__ == '__main__':
    log_results = "./log/test/results/"
    print("Loading data...")
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings()

    print("Intitializing agent...")
    lr = 0.0001
    agent = Agent(G1_adj_matrix, G2_adj_matrix,
                emb1.shape[1], 128, 64, gamma=0.99, activation='Sigmoid')
    agent.load_state_dict(torch.load("./log/train/weights/best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_embeddings_torch = torch.from_numpy(
        emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        emb2).type(torch.FloatTensor).to(device)
    agent.to(device)
    agent.eval()
    total_match = 0
    for k, v in tqdm(ground_truth.items()):
        
        action, p = agent.get_action(first_embeddings_torch,
                                   second_embeddings_torch, [(k,v)], False)
        if action == 1:
            total_match += 1
    
    accuracy = total_match/len(ground_truth)
    print("Accuracy: ", accuracy)
    save_results([accuracy], log_results + "accuracy.csv")
