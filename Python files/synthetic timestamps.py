import random
import networkx as nx
import pickle
import numpy as np
from datetime import datetime, timedelta

# Load Amazon Graph
with open("processed_amazon_graph.pkl", "rb") as f:
    amazon_graph = pickle.load(f)

# Define a reasonable time range for synthetic timestamps
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)

# Generate timestamps using a Gaussian distribution (most edges between 2015-2017)
mean_year = (2015 - 2010) / (2020 - 2010)  # Center around 2015-2017
std_dev = 0.15  # Control spread

for u, v in amazon_graph.edges():
    # Generate a timestamp in range [0,1] (normal distribution)
    normalized_time = np.clip(np.random.normal(mean_year, std_dev), 0, 1)

    # Convert to actual datetime
    random_days = int(normalized_time * (end_date - start_date).days)
    timestamp = (start_date + timedelta(days=random_days)).timestamp()

    # Assign timestamp to edge
    amazon_graph[u][v]["timestamp"] = timestamp

# Save the updated graph with timestamps
with open("processed_amazon_graph_with_timestamps.pkl", "wb") as f:
    pickle.dump(amazon_graph, f)

print("✅ Synthetic timestamps added to Amazon dataset and saved!")
