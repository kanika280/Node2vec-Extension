import random
import networkx as nx
import pickle
import numpy as np
from datetime import datetime, timedelta

with open("processed_amazon_graph.pkl", "rb") as f:
    amazon_graph = pickle.load(f)

start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)

mean_year = (2015 - 2010) / (2020 - 2010)  
std_dev = 0.15  

for u, v in amazon_graph.edges():
    normalized_time = np.clip(np.random.normal(mean_year, std_dev), 0, 1)

    random_days = int(normalized_time * (end_date - start_date).days)
    timestamp = (start_date + timedelta(days=random_days)).timestamp()

    amazon_graph[u][v]["timestamp"] = timestamp

with open("processed_amazon_graph_with_timestamps.pkl", "wb") as f:
    pickle.dump(amazon_graph, f)

print("✅ Synthetic timestamps added to Amazon dataset and saved!")
