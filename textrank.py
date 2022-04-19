from typing import List, Tuple
import torch
import torch.nn.functional as f
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/gtr-t5-xxl')

def calc_adjacency(nodes: List[str]) -> torch.tensor:
    vecs = torch.tensor(model.encode(nodes))
    dists = torch.cdist(vecs, vecs)
    sims = (dists.max(dim=0)[0] - dists)
    sims = vecs.matmul(vecs.t())
    return sims / sims.sum(dim=0)

def summorize(text: str) -> Tuple[List[str], List[int]]:
    nodes = sent_tokenize(text)
    adjacency_matrix = calc_adjacency(nodes)
    qd = torch.linalg.eig(adjacency_matrix)
    #print(qd.eigenvalues)
    ranks = np.flip(qd.eigenvectors[0].real.softmax(dim=0).numpy().argsort()).argsort()
    print(ranks, qd.eigenvectors[0].real.softmax(dim=0).numpy())
    
    return ranks, nodes
