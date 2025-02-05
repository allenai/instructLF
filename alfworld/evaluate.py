def evaluate_ours(data_dir='./compile_data/compiled_sequences/valid_seen'):
    import joblib
    import os 
    import numpy as np
    from tqdm import tqdm


    data = []

    for file in os.listdir(data_dir):

        data.append(joblib.load(os.path.join(data_dir, file)))

    def get_trajectory_text(sequence):
        out = []
        for dp in sequence:
            if isinstance(dp, dict):
                out.append({k:v for k,v in dp.items() if k!='admissible_commands'})
            else:
                out.append(dp)
        return str(out)

    def get_teacher_forcing_data(sequence):
        out = []

        for idx, dp in enumerate(sequence):
            if not isinstance(dp, dict): # this is an action
                out.append({'input':get_trajectory_text(sequence[:idx]), 'label': dp})

        return out

    test_cases = []

    for sequence in data:
        traj = get_trajectory_text(sequence)
        teacher_forcing_data = get_teacher_forcing_data(sequence)
        for dp in teacher_forcing_data:
            info = {'cur_state': dp['input'], 'gold_action':dp['label'][1]}
            test_cases.append(info)

    training_raw_matrix = joblib.load('./assets/training_state_concept_embs.joblib')

    import torch 
    import joblib 
    from linear_corex import Corex
    from data_utils import get_training_state_actions, get_unique_concepts

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    train_states, train_actions = get_training_state_actions()
    unique_concepts = get_unique_concepts()

    model = SentenceTransformer('models/minilm-final')
    embeddings = model.encode(unique_concepts, show_progress_bar=True, convert_to_tensor=True)

    cache = dict()

    def to_concept_space(nl):
        if nl not in cache:
            q = model.encode(nl, convert_to_tensor=True)
            cos_similarities = cos_sim(q, embeddings).flatten().cpu()
            cache[nl] = cos_similarities
            return cos_similarities
        else:
            return cache[nl]



    test_states = [tc['cur_state'] for tc in test_cases]
    test_actions = [tc['gold_action'] for tc in test_cases]
    test_states_emb = torch.stack([to_concept_space(s) for s in test_states]).numpy()

    from sentence_transformers.util import cos_sim
    corex_bundle = joblib.load('./assets/corex.joblib')

    def replace_digits_with_space(s):
        """
        This function takes a string and replaces all digits with empty spaces.
        
        Parameters:
        s (str): The input string.
        
        Returns:
        str: The modified string with digits replaced by empty spaces.
        """
        return ''.join(' ' if char.isdigit() else char for char in s)

    def ignore_digits(lst):
        out = []
        for inp in lst:
            out.append(replace_digits_with_space(inp))
        return out

    for i in range(1,len(corex_bundle['dims'])+1):
        print('Using corex layers: ', corex_bundle['dims'][:i])
        cur_train_embs = training_raw_matrix 
        for layer in corex_bundle['corex_layers'][:i]:
            cur_train_embs = layer.transform(cur_train_embs)

        pred_actions = []
        for te in tqdm(test_states_emb):
            q = te 
            for layer in corex_bundle['corex_layers'][:i]:
                q = layer.transform(q)
            scores = cos_sim(q, cur_train_embs).flatten()
            ranking = np.argsort(-scores)
            selected_action = train_actions[ranking[0]]
            pred_actions.append(selected_action)
        tot_correct = sum(np.array(ignore_digits(pred_actions)) == np.array(ignore_digits(test_actions)))
        score = tot_correct/len(test_states)
        print(tot_correct, score)

if __name__ == '__main__':
    print('seen: ')
    evaluate_ours(data_dir='./compile_data/compiled_sequences/valid_seen')
    print('unseen: ')
    evaluate_ours(data_dir='./compile_data/compiled_sequences/valid_unseen')
