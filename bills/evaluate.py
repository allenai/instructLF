import pandas as pd
raw_annotations = pd.read_csv('./assets/bills_annotations.csv')
from data_utils import parse

train_docs = raw_annotations['texts']
train_docs[25]
all_concepts = []

for anno in raw_annotations['concepts']:
    c = parse(anno)
    all_concepts+= c

from data_utils import get_unique_concepts
unique_concepts = get_unique_concepts()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models/minilm-final')

embeddings = model.encode(unique_concepts, show_progress_bar=True, convert_to_tensor=True)

def compute_feature_cos_sims(nl):
    q = model.encode(nl, convert_to_tensor=True)
    cos_similarities = cos_sim(q, embeddings).flatten().cpu()
    return cos_similarities

from linear_corex import Corex
import joblib
corex_bundle = joblib.load('assets/corex.joblib')

corex_layer = corex_bundle['corex_layers'][0]

clusters_ids = corex_layer.clusters()

from collections import defaultdict
clusters = defaultdict(list)
for cid, concept in zip(clusters_ids, unique_concepts):
    clusters[cid].append(concept)
clusters = dict(clusters)

cluster_items = list(clusters.values())

import json
with open('data/topic_assignment.jsonl', 'r') as ifp:
    test_data = [json.loads(i.strip()) for i in ifp.readlines()]

test_docs = [i['summary'] for i in test_data]

from tqdm import tqdm
from sentence_transformers.util import cos_sim 

test_embs  = []
for doc in tqdm(test_docs): 
    test_embs.append(compute_feature_cos_sims(doc))

import numpy as np
test_embs_transformed = corex_layer.transform(np.array([i.tolist() for i in test_embs]))


from collections import Counter
test_labels = [i['topic'] for i in test_data]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
test_labels_transformed = le.fit_transform(test_labels)
X_train, X_test, y_train, y_test = train_test_split(test_embs_transformed, test_labels_transformed,
                                                    stratify=test_labels, 
                                                    test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_pred = y_pred, y_true=y_test)

pop_y_pred =[Counter(y_train).most_common(1)[0][0] for i in range(len(y_test))]

balanced_accuracy_score(y_pred = pop_y_pred, y_true=y_test)

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
import numpy as np
from collections import Counter

# Transform labels
le = LabelEncoder()
test_labels_transformed = le.fit_transform(test_labels)

# Define the classifier
# clf = LogisticRegression(max_iter=1000)
clf = DecisionTreeClassifier()

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5)

# Define a custom scoring function
scorer = make_scorer(balanced_accuracy_score)

# Perform cross-validation
scores = cross_val_score(clf, test_embs_transformed, test_labels_transformed, cv=cv, scoring=scorer)

# Print results
print("Balanced Accuracy Scores for each fold:")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")

print(f"\nAverage Balanced Accuracy Score: {np.mean(scores):.4f}")

# Compute and print performance of the most common class prediction (for comparison)
pop_y_pred = [Counter(test_labels_transformed).most_common(1)[0][0] for _ in range(len(test_labels_transformed))]
pop_score = balanced_accuracy_score(pop_y_pred, test_labels_transformed)

print(f"\nBalanced Accuracy Score using the most common class: {pop_score:.4f}")


from collections import Counter
test_labels = [i['subtopic'] for i in test_data]

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score
import numpy as np
from collections import Counter

# Transform labels
le = LabelEncoder()
test_labels_transformed = le.fit_transform(test_labels)

# Define the classifier
# clf = LogisticRegression(max_iter=1000)
clf = DecisionTreeClassifier()

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5)

# Define a custom scoring function
scorer = make_scorer(balanced_accuracy_score)

# Perform cross-validation
scores = cross_val_score(clf, test_embs_transformed, test_labels_transformed, cv=cv, scoring=scorer)

# Print results
print("Balanced Accuracy Scores for each fold:")
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.4f}")

print(f"\nAverage Balanced Accuracy Score: {np.mean(scores):.4f}")

# Compute and print performance of the most common class prediction (for comparison)
pop_y_pred = [Counter(test_labels_transformed).most_common(1)[0][0] for _ in range(len(test_labels_transformed))]
pop_score = balanced_accuracy_score(pop_y_pred, test_labels_transformed)

print(f"\nBalanced Accuracy Score using the most common class: {pop_score:.4f}")



