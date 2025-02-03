import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load Amazon embeddings (format: node_id, embedding values)
amazon_embeddings = {}
with open("temporal_node2vec_embeddings_amazon.txt", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:  # Skip first line (metadata)
        parts = line.strip().split()
        node_id = int(parts[0])  # First value is node ID
        embedding = np.array([float(x) for x in parts[1:]])  # Remaining values are embedding
        amazon_embeddings[node_id] = embedding

# Convert to numpy array
node_ids = list(amazon_embeddings.keys())
embeddings_matrix = np.array([amazon_embeddings[node] for node in node_ids])

# Cluster nodes using K-Means (You can adjust the number of clusters)
num_clusters = 10  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings_matrix)

# Assign cluster labels
amazon_labels = {node: [int(clusters[i])] for i, node in enumerate(node_ids)}

# Save labels
with open("amazon_labels.pkl", "wb") as f:
    pickle.dump(amazon_labels, f)

print(f"✅ Amazon Clustering Done! {num_clusters} clusters assigned. Labels saved in 'amazon_labels.pkl'.")
