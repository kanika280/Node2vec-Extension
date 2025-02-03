from datetime import datetime
import networkx as nx
import numpy as np

def load_arxiv_graph(arxiv_edges_filepath, arxiv_dates_filepath):
    arxiv_graph = nx.DiGraph()
    dates = {}

    # Load timestamps
    print("Loading timestamps...")
    with open(arxiv_dates_filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                node, date_str = parts
                try:
                    timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
                    dates[int(node)] = timestamp
                except ValueError:
                    print(f"Skipping malformed date entry: {line.strip()}")

    print(f"Loaded {len(dates)} timestamps.")

    # Load edges
    print("Loading edges...")
    with open(arxiv_edges_filepath, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comment/header lines
                continue
            try:
                source, target = map(int, line.strip().split())  # Space-separated
                arxiv_graph.add_edge(source, target)
            except ValueError:
                print(f"Skipping invalid edge line: {line.strip()}")

    print(f"Graph loaded with {arxiv_graph.number_of_nodes()} nodes and {arxiv_graph.number_of_edges()} edges.")

    # Assign timestamps
    missing_nodes = 0
    for node in arxiv_graph.nodes():
        if node in dates:
            arxiv_graph.nodes[node]["timestamp"] = dates[node]
        else:
            arxiv_graph.nodes[node]["timestamp"] = -1  # Placeholder for missing nodes
            missing_nodes += 1

    print(f"Nodes still missing timestamps after assignment: {missing_nodes}/{len(arxiv_graph.nodes())}")

    # Fill missing timestamps based on neighbors
    print("Filling missing timestamps...")
    all_timestamps = [ts for ts in dates.values()]
    median_timestamp = int(np.median(all_timestamps)) if all_timestamps else -1

    filled_count = 0
    for node in arxiv_graph.nodes():
        if arxiv_graph.nodes[node]["timestamp"] == -1:  # Missing timestamp
            neighbor_timestamps = [
                arxiv_graph.nodes[n]["timestamp"]
                for n in arxiv_graph.neighbors(node)
                if arxiv_graph.nodes[n]["timestamp"] != -1
            ]
            if neighbor_timestamps:
                arxiv_graph.nodes[node]["timestamp"] = int(np.mean(neighbor_timestamps))
            else:
                arxiv_graph.nodes[node]["timestamp"] = median_timestamp  # Assign median if no neighbors have timestamps
            filled_count += 1

    print(f"Filled timestamps for {filled_count} nodes.")
    print(f"Nodes still missing timestamps after filling: {sum(1 for n in arxiv_graph.nodes() if arxiv_graph.nodes[n]['timestamp'] == -1)}")

    return arxiv_graph

# File paths
arxiv_edges_filepath = "Cit-HepTh.txt"
arxiv_dates_filepath = "Cit-HepTh-dates.txt"

# Load and process graph
arxiv_graph = load_arxiv_graph(arxiv_edges_filepath, arxiv_dates_filepath)

# Print first few nodes
print(list(arxiv_graph.nodes(data=True))[:10])
