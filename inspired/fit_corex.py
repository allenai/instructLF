def fit_corex(corex_hidden=[10,6,4]):

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('models/minilm-final')

    from linear_corex import Corex

    from data_utils import get_unique_concepts, get_training_states
    unique_concepts = get_unique_concepts()
    states = get_training_states()
    embeddings = model.encode(unique_concepts, show_progress_bar=True, convert_to_tensor=True)
    embeddings = model.encode(
        unique_concepts, 
        show_progress_bar=True, 
        convert_to_tensor=True
        )
    cache = dict()

    def to_concept_space(nl):
        if nl not in cache:
            q = model.encode(nl, convert_to_tensor=True)
            cos_similarities = cos_sim(q, embeddings).flatten().cpu()
            cache[nl] = cos_similarities
            return cos_similarities
        else:
            return cache[nl]
        
    from tqdm import tqdm
    from sentence_transformers.util import cos_sim

    for state in tqdm(states):
        to_concept_space(state)

    import numpy as np
    import torch

    X_train = torch.stack([to_concept_space(s) for s in states]).numpy()

    import joblib 
    joblib.dump(X_train, './assets/training_state_concept_embs.joblib')

    corex_layer_dims = [len(unique_concepts)] + corex_hidden 
    corex_layers = []
    layer_ios = []
    for i in range(len(corex_layer_dims)-1):
        layer_io = (corex_layer_dims[i], corex_layer_dims[i+1])
        print('Adding corex layer: ', layer_io)
        layer = Corex( nv=layer_io[0], n_hidden=layer_io[1], max_iter=1000, tol=1e-5, anneal=True, missing_values=None,
                    gaussianize='standard', l1=0.9, device='cuda:0', stopping_len=50, verbose=0,
                    optimizer_class=torch.optim.Adam, optimizer_params={})
        corex_layers.append(layer)
        layer_ios.append(layer_io)

    cur_data = X_train
    for i, layer in enumerate(corex_layers):
        print('fitting layers with dim: ', layer_ios[i])
        layer.fit(cur_data)
        print('done fitting')
        cur_data = layer.transform(cur_data)

    import joblib 
    joblib.dump({'corex_layers':corex_layers, 'dims': layer_ios}, './assets/corex.joblib')

if __name__ == '__main__':
    fit_corex(corex_hidden=[368])