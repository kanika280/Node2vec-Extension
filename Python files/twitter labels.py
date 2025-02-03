import json
import pickle

# Load Twitter dataset (Congress network)
with open("congress_network_data.json", "r") as f:
    data = json.load(f)

# Extract labels from "inList"
node_labels = {}  # Dictionary to store {node_id: [labels]}
for label, nodes in enumerate(data[0]["inList"]):  # Assuming "inList" is at index 0
    for node in nodes:
        if node not in node_labels:
            node_labels[node] = []
        node_labels[node].append(label)  # Assign multiple labels if in multiple groups

# Save extracted labels
with open("twitter_labels.pkl", "wb") as f:
    pickle.dump(node_labels, f)

print(f"✅ Extracted labels for {len(node_labels)} nodes. Labels saved in 'twitter_labels.pkl'.")
