import numpy as np
import pandas as pd
import re

def parse(annotation):
    premises = annotation

    pattern = r'\d+\.\s'
    matches = re.findall(pattern, premises)
    matches = [match.strip() for match in matches]

    
    for m in matches:
        premises = premises.replace(m, '<split>')

    concepts = [i.strip().lower().lstrip('when ') for i in premises.split('<split>')]

    return [c for c in concepts[1:]]

def get_training_state_actions():
    raw_annotations = pd.read_csv('./assets/alfworld_annotations.csv')
    states = list(raw_annotations['states'])
    actions = list(raw_annotations['gold_actions'])
    return states, actions

def get_unique_concepts():

    raw_annotations = pd.read_csv('./assets/alfworld_annotations.csv')
    states = list(raw_annotations['states'])
    actions = list(raw_annotations['gold_actions'])
    len(states), len(actions)

    all_concepts = []

    for anno in raw_annotations['concepts']:
        c = parse(anno)
        all_concepts+= c

    unique_concepts = [c for c in sorted(list(set(all_concepts))) if len(c) > 0]
    unique_concepts = np.array(unique_concepts)

    return unique_concepts


if __name__ == "__main__":
    pass