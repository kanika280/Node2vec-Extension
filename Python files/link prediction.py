import random
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors

with open("processed_arxiv_graph.pkl", "rb") as f:
    graph = pickle.load(f)

embedding_model = KeyedVectors.load_word2vec_format("temporal_node2vec_embeddings.txt", binary=False)

edges = list(graph.edges(data=True))

edges = sorted(edges, key=lambda x: x[2].get('timestamp', 0))

train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)

nodes = list(graph.nodes())
non_edges = set()
while len(non_edges) < len(test_edges):
    u, v = random.sample(nodes, 2)
    if not graph.has_edge(u, v):
        non_edges.add((u, v))
test_negative_edges = list(non_edges)

def cosine_similarity(node1, node2):
    if str(node1) in embedding_model and str(node2) in embedding_model:
        return embedding_model.similarity(str(node1), str(node2))
    else:
        return 0 

y_true = [1] * len(test_edges) + [0] * len(test_negative_edges)
y_scores = [cosine_similarity(u, v) for u, v, _ in test_edges] + [cosine_similarity(u, v) for u, v in test_negative_edges]

# Evaluate using AUC (Area Under Curve)
auc_score = roc_auc_score(y_true, y_scores)
print(f"🔥 **Temporal Link Prediction AUC Score: {auc_score:.4f}**")
