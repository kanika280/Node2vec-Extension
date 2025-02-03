import gensim
from gensim.models import Word2Vec

# Function to train and save embeddings
def train_skip_gram(dataset_name, walks_file, embedding_output):
    print(f"🚀 Training Skip-Gram for {dataset_name}...")

    # Load walks from file
    with open(walks_file, "r") as f:
        walks = [line.strip().split() for line in f.readlines()]

    # Train Word2Vec model (Skip-Gram)
    model = Word2Vec(sentences=walks, vector_size=128, window=5, min_count=1, sg=1, workers=4, epochs=10)

    # Save embeddings
    model.wv.save_word2vec_format(embedding_output, binary=False)
    print(f"✅ Skip-Gram training completed for {dataset_name}!")
    print(f"📂 Embeddings saved as {embedding_output}")

# Train on Amazon dataset
train_skip_gram("Amazon", "temporal_walks_amazon.txt", "temporal_node2vec_embeddings_amazon.txt")

# Train on Twitter dataset
train_skip_gram("Twitter", "temporal_walks_twitter.txt", "temporal_node2vec_embeddings_twitter.txt")

print("🎯 All Skip-Gram models trained successfully!")
