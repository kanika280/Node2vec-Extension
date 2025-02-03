import gensim
from gensim.models import Word2Vec

def train_skip_gram(dataset_name, walks_file, embedding_output):
    print(f"🚀 Training Skip-Gram for {dataset_name}...")

    with open(walks_file, "r") as f:
        walks = [line.strip().split() for line in f.readlines()]

    model = Word2Vec(sentences=walks, vector_size=128, window=5, min_count=1, sg=1, workers=4, epochs=10)

    model.wv.save_word2vec_format(embedding_output, binary=False)
    print(f"✅ Skip-Gram training completed for {dataset_name}!")
    print(f"📂 Embeddings saved as {embedding_output}")

train_skip_gram("Amazon", "temporal_walks_amazon.txt", "temporal_node2vec_embeddings_amazon.txt")

train_skip_gram("Twitter", "temporal_walks_twitter.txt", "temporal_node2vec_embeddings_twitter.txt")

print("🎯 All Skip-Gram models trained successfully!")
