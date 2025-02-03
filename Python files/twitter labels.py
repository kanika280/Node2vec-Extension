import json
import pickle

with open("congress_network_data.json", "r") as f:
    data = json.load(f)

node_labels = {} 
for label, nodes in enumerate(data[0]["inList"]): 
    for node in nodes:
        if node not in node_labels:
            node_labels[node] = []
        node_labels[node].append(label)  

with open("twitter_labels.pkl", "wb") as f:
    pickle.dump(node_labels, f)

print(f"✅ Extracted labels for {len(node_labels)} nodes. Labels saved in 'twitter_labels.pkl'.")
