import pandas as pd

annotations = pd.read_csv('./assets/bills_annotations.csv')
features = list(annotations['concepts'])
queries = annotations['texts']

import re

def parse(notes):
    # Use regex to find the patterns
    pattern = r'\d+\.\s'
    matches = re.findall(pattern, notes)
    matches = [match.strip() for match in matches]
    processed_notes = notes
    for m in matches:
        processed_notes = processed_notes.replace(m, '<split>')
    concepts = processed_notes.split('<split>')
    
    return concepts[1:]

parsed_features = []
for f in features:
    parsed_features.append(parse(f))

positive_samples = []
for q, feats in zip(queries, parsed_features):
    for f in feats:
        positive_samples.append( (q, f) )

print('sample pairs')
print(positive_samples[5])

possible_feats = sorted(list(set([i[1] for i  in positive_samples])))
print('sample concepts')
print(possible_feats[50:80])

samples = positive_samples 

from sklearn.model_selection import train_test_split

# Split data into training, validation, and test sets
train_samples, test_samples = train_test_split(samples, test_size=0.1, random_state=42)
val_samples, test_samples = train_test_split(test_samples, test_size=0.005, random_state=42)

import datasets
train_dataset = datasets.Dataset.from_dict({
    'premise': [i[0] for i in train_samples],
    'hypothesis': [i[1] for i in train_samples],
})

eval_dataset = datasets.Dataset.from_dict({
    'premise': [i[0] for i in val_samples],
    'hypothesis': [i[1] for i in val_samples],
})

test_dataset = datasets.Dataset.from_dict({
    'premise': [i[0] for i in test_samples],
    'hypothesis': [i[1] for i in test_samples],
})

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
loss = MultipleNegativesRankingLoss(model)


from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/minilm-inspired",
    # Optional training parameters:
    num_train_epochs=20,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    warmup_ratio=0.1,
    # Optional tracking/debugging parameters:
    save_strategy="epoch",
    evaluation_strategy='epoch',
    save_total_limit=1,
    learning_rate=5e-5,
    load_best_model_at_end = True
)

from sentence_transformers.evaluation import BinaryClassificationEvaluator



from sentence_transformers import SentenceTransformerTrainer

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss
)

from transformers import EarlyStoppingCallback

early_stop = EarlyStoppingCallback(early_stopping_patience=2)

trainer.add_callback(early_stop)

trainer.train()


model.save_pretrained("models/minilm-final")