import pandas as pd
import re

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

def get_unique_concepts():
    annotations = pd.read_csv('./assets/bills_annotations.csv')
    features = list(annotations['concepts'])

    parsed_features = []
    for f in features:
        parsed_features.append(parse(f))

    import itertools

    possible_feats = sorted(list(set(itertools.chain(*parsed_features))))
    return possible_feats

def get_training_states():
    annotations = pd.read_csv('./assets/bills_annotations.csv')
    queries = annotations['texts']
    return queries

