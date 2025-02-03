import itertools


def generate_skipgram_pairs(walks, window_size=5):
    """
    Converts sequences of temporal random walks into skip-gram training pairs.

    :param walks: List of random walk sequences
    :param window_size: Context window size
    :return: List of (node, context) training pairs
    """
    pairs = []
    for walk in walks:
        for i, target in enumerate(walk):
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            context_nodes = walk[start:i] + walk[i + 1:end]

            pairs.extend([(target, context) for context in context_nodes])

    return pairs


walks = []
with open("temporal_walks.txt", "r") as f:
    for line in f:
        walks.append(line.strip().split())

skipgram_pairs = generate_skipgram_pairs(walks, window_size=5)

with open("skipgram_pairs.txt", "w") as f:
    for target, context in skipgram_pairs:
        f.write(f"{target} {context}\n")

print(f"✅ Skip-gram training data generated: {len(skipgram_pairs)} pairs")

from gensim.models import Word2Vec

skipgram_pairs = []
with open("skipgram_pairs.txt", "r") as f:
    for line in f:
        target, context = line.strip().split()
        skipgram_pairs.append((target, context))

sentences = []
current_sentence = []
for target, context in skipgram_pairs:
    if len(current_sentence) < 10:
        current_sentence.append(context)
    else:
        sentences.append(current_sentence)
        current_sentence = [context]

print("🚀 Training Word2Vec model...")
model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, sg=1, workers=4, epochs=10)

model.wv.save_word2vec_format("temporal_node2vec_embeddings.txt")
print("✅ Temporal Node2Vec embeddings saved!")
