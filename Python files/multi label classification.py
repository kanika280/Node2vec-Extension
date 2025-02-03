import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from gensim.models import KeyedVectors


# 🚀 Load Node Embeddings
def load_embeddings(file_path):
    """Loads Node2Vec embeddings from a text file."""
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)
    embeddings = {int(node): model[node] for node in model.index_to_key}
    return embeddings


# 🚀 Load Node Labels
def load_labels(dataset_name):
    """Loads node labels from a pickle file."""
    with open(f"{dataset_name}_labels.pkl", "rb") as f:
        return pickle.load(f)  # Dictionary {node_id: [label1, label2, ...]}


# 🔥 Train & Evaluate Multi-Label Classification
def train_and_evaluate(embeddings, labels, dataset_name):
    print(f"\n🔍 Running Multi-Label Classification for {dataset_name}...")

    # Align embeddings and labels
    nodes = list(set(embeddings.keys()) & set(labels.keys()))
    X = np.array([embeddings[node] for node in nodes])
    Y = [labels[node] for node in nodes]

    # Convert labels to binary format
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)

    # Split data into train & test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 🔥 Logistic Regression (One-vs-Rest)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    # Alternative: 🔥 Random Forest (More robust for structured data)
    # clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

    clf.fit(X_train, Y_train)

    # Predict
    Y_pred = clf.predict(X_test)
    Y_pred_proba = clf.predict_proba(X_test)

    # Evaluate
    f1 = f1_score(Y_test, Y_pred, average="micro")
    auc = roc_auc_score(Y_test, Y_pred_proba, average="micro")

    print(f"🎯 **{dataset_name} Multi-Label Classification Results:**")
    print(f"🔥 F1-score: {f1:.4f}")
    print(f"🔥 AUC-ROC Score: {auc:.4f}")


# 🚀 Run for Amazon & Twitter
for dataset in ["amazon", "twitter"]:
    embeddings = load_embeddings(f"temporal_node2vec_embeddings_{dataset}.txt")
    labels = load_labels(dataset)
    train_and_evaluate(embeddings, labels, dataset)
