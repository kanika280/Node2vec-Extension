import random
import networkx as nx
import numpy as np

class TemporalRandomWalker:
    def __init__(self, graph, walk_length=10, num_walks=5, decay_factor=0.9):
        """
        :param graph: NetworkX graph (with 'timestamp' attribute for nodes)
        :param walk_length: Length of each random walk
        :param num_walks: Number of walks per node
        :param decay_factor: Weight decay factor for older connections (0 < decay <= 1)
        """
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.decay_factor = decay_factor

    def time_weighted_probabilities(self, neighbors, timestamps, current_time):
        """
        Assigns probabilities based on recency using an exponential decay function.
        """
        time_diffs = np.array([timestamps[n] - current_time for n in neighbors])
        valid_indices = time_diffs >= 0  

        if not np.any(valid_indices):  
            return np.ones(len(neighbors)) / len(neighbors)

        decay_weights = np.exp(-self.decay_factor * time_diffs[valid_indices]) 

        if np.sum(decay_weights) == 0:  
            return np.ones(len(neighbors)) / len(neighbors)

        decay_weights /= np.sum(decay_weights) 

        probs = np.zeros(len(neighbors))
        probs[valid_indices] = decay_weights
        return probs

    def temporal_random_walk(self, start_node):
        """
        Performs a single temporal random walk.
        """
        walk = [start_node]
        current_time = self.graph.nodes[start_node].get('timestamp', float('-inf'))


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
        """
        Generates multiple random walks for each node.
        """
        walks = []
        for node in self.graph.nodes():
            for _ in range(self.num_walks):
                walks.append(self.temporal_random_walk(node))
        return walks


if __name__ == "__main__":
  
    import pickle

    with open("processed_arxiv_graph.pkl", "rb") as f:
        arxiv_graph = pickle.load(f)

    walker = TemporalRandomWalker(arxiv_graph, walk_length=10, num_walks=5, decay_factor=0.9)
    walks = walker.generate_walks()

    with open("temporal_walks.txt", "w") as f:
        for walk in walks:
            f.write(" ".join(map(str, walk)) + "\n")

    print("✅ Temporal random walks generated and saved!")


