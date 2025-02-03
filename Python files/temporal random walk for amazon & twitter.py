import random
import networkx as nx
import numpy as np
import pickle


class TemporalRandomWalker:
    def __init__(self, graph, dataset_name, walk_length=10, num_walks=5, decay_factor=0.9):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.decay_factor = decay_factor
        self.dataset_name = dataset_name

    def time_weighted_probabilities(self, neighbors, timestamps, current_time):
        time_diffs = np.array([timestamps[n] - current_time for n in neighbors])
        valid_indices = time_diffs >= 0  # Only allow forward-time traversal

        if not np.any(valid_indices):
            return np.ones(len(neighbors)) / len(neighbors)

        decay_weights = np.exp(-self.decay_factor * time_diffs[valid_indices])

        if np.sum(decay_weights) == 0:  # Prevent division by zero
            return np.ones(len(neighbors)) / len(neighbors)

        decay_weights /= np.sum(decay_weights)  # Normalize probabilities

        probs = np.zeros(len(neighbors))
        probs[valid_indices] = decay_weights

        # **Fix NaN Issue**: Replace NaNs with uniform probabilities
        if np.isnan(probs).any():
            probs = np.ones(len(neighbors)) / len(neighbors)

        return probs

    def temporal_random_walk(self, start_node):
        walk = [start_node]
        current_time = self.graph.nodes[start_node].get('timestamp', 0)  # Default timestamp if missing

        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(walk[-1]))
            if not neighbors:
                break

            timestamps = {n: self.graph.nodes[n].get('timestamp', float('inf')) for n in neighbors}
            probs = self.time_weighted_probabilities(neighbors, timestamps, current_time)

            next_node = np.random.choice(neighbors, p=probs)
            walk.append(next_node)
            current_time = timestamps[next_node]

        return walk

    def generate_walks(self):
        walks = []
        for node in self.graph.nodes():
            for _ in range(self.num_walks):
                walks.append(self.temporal_random_walk(node))

        # Save walks
        with open(f"temporal_walks_{self.dataset_name}.txt", "w") as f:
            for walk in walks:
                f.write(" ".join(map(str, walk)) + "\n")

        print(f"✅ Temporal random walks generated and saved for {self.dataset_name}!")


# Process Amazon & Twitter
datasets = {
    "amazon": "processed_amazon_graph_with_timestamps.pkl",
    "twitter": "processed_twitter_graph.pkl",
}

for dataset, file_path in datasets.items():
    try:
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        walker = TemporalRandomWalker(graph, dataset, walk_length=10, num_walks=5, decay_factor=0.9)
        walker.generate_walks()
    except FileNotFoundError:
        print(f"❌ Processed graph missing for {dataset.upper()}! Skipping...")
