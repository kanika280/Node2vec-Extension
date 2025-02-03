import networkx as nx
import pandas as pd
from datetime import datetime

# Load Amazon Dataset (Undirected Graph)
def load_amazon_graph(filepath):
    G = nx.read_edgelist(filepath, delimiter="\t", nodetype=int, create_using=nx.Graph())
    print(f"Amazon Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# Load ArXiv Citation Network (Directed Graph with Timestamps)
def load_arxiv_graph(edges_filepath, dates_filepath):
    G = nx.DiGraph()

    # Load edges
    with open(edges_filepath, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)

    # Load timestamps (Converting date to Unix timestamp)
    dates = {}
    with open(dates_filepath, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comments
                continue
            node, date_str = line.strip().split()
            try:
                # Convert 'YYYY-MM-DD' to Unix timestamp
                timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
                dates[int(node)] = timestamp
            except ValueError as e:
                print(f"Error parsing date {date_str} for node {node}: {e}")

    nx.set_node_attributes(G, dates, "timestamp")
    print(f"ArXiv Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# Load Twitter (Congress) Dataset (Undirected Graph)
def load_twitter_graph(filepath):
    G = nx.read_edgelist(filepath, nodetype=int, create_using=nx.Graph())
    print(f"Twitter Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# File Paths (Modify these paths based on where you've stored your files)
amazon_filepath = "com-amazonungraph.txt"
arxiv_edges_filepath = "cit-HepTh.txt"
arxiv_dates_filepath = "cit-HepTh-dates.txt"
twitter_filepath = "congress.edgelist"

# Load Graphs
amazon_graph = load_amazon_graph(amazon_filepath)
arxiv_graph = load_arxiv_graph(arxiv_edges_filepath, arxiv_dates_filepath)
twitter_graph = load_twitter_graph(twitter_filepath)

# Check first 5 nodes with timestamps
print(list(arxiv_graph.nodes(data=True))[:5])

missing_timestamps = sum(1 for _, data in arxiv_graph.nodes(data=True) if "timestamp" not in data)
print(f"Nodes missing timestamps: {missing_timestamps}/{arxiv_graph.number_of_nodes()}")



import pickle

# Save the processed Arxiv graph as a pickle file
with open("processed_arxiv_graph.pkl", "wb") as f:
    pickle.dump(arxiv_graph, f)

print("✅ Processed Arxiv graph saved as 'processed_arxiv_graph.pkl'")

# Save processed graphs
with open("processed_amazon_graph.pkl", "wb") as f:
    pickle.dump(amazon_graph, f)

with open("processed_twitter_graph.pkl", "wb") as f:
    pickle.dump(twitter_graph, f)

print("✅ Amazon & Twitter Graphs Processed and Saved!")
