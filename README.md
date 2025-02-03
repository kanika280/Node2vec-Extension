Node2Vec-Extension: Temporal and Dynamic Network Analysis
Extending Node2Vec for Time-Aware Graph Embeddings

This repository contains an extension of Node2Vec, incorporating temporal dynamics to improve link prediction and multi-label classification in evolving graphs. Traditional node2vec assumes a static graph structure, but real-world networks change over time. This work modifies node2vec’s random walks to respect chronological order and introduces a decay factor, prioritizing recent interactions.

Overview of the Extension
This implementation extends node2vec to include temporal constraints, making it more effective for:
-Temporal Link Prediction → Predicts future edges based on historical interactions.
-Multi-Label Classification → Assigns evolving categories to nodes in dynamic networks.

Key Modifications
-Temporal Random Walks → Ensures walks respect chronological order of edges.
-Decay Factor → Prioritizes recent interactions over older ones.
-Skip-gram Training with Time-Aware Walks → Generates more relevant embeddings for evolving graphs.
-Scalability → Maintains the efficiency of node2vec while incorporating temporal aspects.

Datasets Used
Since original node2vec datasets lack timestamps, we used:

-ArXiv HEP-TH (Citation Network) → For Temporal Link Prediction.
-Amazon Product Co-Purchase Graph → For Multi-Label Classification.
-Twitter Congress Network → For Multi-Label Classification.
