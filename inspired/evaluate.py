from data_utils import get_unique_concepts, get_training_states, get_inspired_data, get_inspired_data_with_heldout
import joblib

corex_bundle = joblib.load('./assets/corex.joblib')

def corex_transform():
    # dummy function to be used later
    pass 


training_data, movie_vocab = get_inspired_data(
    './datasets/inspired_train.csv'
)

queries = get_training_states()

concept_embeddings_train = joblib.load('./assets/training_state_concept_embs.joblib')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('models/minilm-inspired-final')

from data_utils import get_unique_concepts, get_training_states
import torch 
unique_concepts = get_unique_concepts()
states = get_training_states()
embeddings = model.encode(unique_concepts, show_progress_bar=True, convert_to_tensor=True)
emb_cache = dict()

for idx, state in enumerate(states):
    emb_cache[states[idx]] = torch.from_numpy(concept_embeddings_train[idx])

def to_concept_space(nl):
    if nl not in emb_cache:
        q = model.encode(nl, convert_to_tensor=True)
        cos_similarities = cos_sim(q, embeddings).flatten().cpu()
        emb_cache[nl] = cos_similarities
        return cos_similarities
    else:
        return emb_cache[nl]
    
from tqdm import tqdm
import torch
from sentence_transformers.util import cos_sim

def build_datastore_embedding(sentences):
    
    # Initialize an empty list to hold the sentence embeddings
    all_sentence_embeddings = []
    
    for s in sentences:
        all_sentence_embeddings.append(to_concept_space(s))

    return torch.from_numpy(corex_transform(torch.stack(all_sentence_embeddings).numpy()))

from collections import defaultdict

class KNNLMForTuningHyperParams:

    def __init__(
        self, 
        data_path,
        device = 'cuda:0',
        heldout_portion = 0.2
    ):
        
        training_dataset, validation_dataset, movie_vocab = get_inspired_data_with_heldout(data_path, heldout_portion=heldout_portion)
        movie_vocab = [m.split(' (')[0] for m in movie_vocab] # remove the year at end
        training_posts = sorted(list(set(training_dataset['context'])))
        trainingpost2idx = dict(zip(
            training_posts, 
            list(range(len(training_posts)))
        ))

        trainingpostidx2movies = defaultdict(list)
        movie_from_posts = []
        for i in range(len(training_dataset['context'])):
            post_idx = trainingpost2idx[training_dataset['context'][i]]
            # the split is for removing year at end, e.g. " (2019)"
            # movie = movie_vocab[training_dataset['label'][i]].split(' (')[0] 
            trainingpostidx2movies[post_idx].append(training_dataset['label'][i])

        post_embeddings = build_datastore_embedding(
            sentences = training_posts
        )
        self.device = device
        self.movie_vocab = movie_vocab
        self.training_posts = training_posts
        self.post_embeddings = post_embeddings.to(self.device)
        self.trainingpostidx2movies = trainingpostidx2movies
        self.validation_dataset = validation_dataset

    def encode_sentences(self, batch_sentences):
        return build_datastore_embedding(batch_sentences).to(self.device)

    def top_post_ids_retrieval(self, query):
    
        query_embedding = query# self.encode_sentences(query)[0].reshape(1, -1)
        cosine_similarities = torch.cosine_similarity(query_embedding, self.post_embeddings).cpu().numpy()
        sorted_indices = np.argsort(cosine_similarities)[::-1]
        return sorted_indices, cosine_similarities[sorted_indices]

    def count_based_probability(
        self,
        query, 
        num_posts_to_consider=30, 
        return_logits=False, 
        distance_weighting=False,
        temperature = 1.0
    ):
        relevant_post_ids, similarities = self.top_post_ids_retrieval(query)
        movie_pool = defaultdict(int)
        probas = np.zeros(len(self.movie_vocab))
        for i, id in enumerate(relevant_post_ids[:num_posts_to_consider]):
            for movieid in self.trainingpostidx2movies[id]:
                if distance_weighting:
                    probas[movieid] += similarities[i]/temperature
                else:
                    probas[movieid] += 1
        if return_logits:
            return probas
        probas = torch.nn.functional.softmax(torch.from_numpy(probas), dim=-1)
        return probas
    

from data_utils import get_inspired_data_with_heldout
import numpy as np
import pandas as pd
from scipy.stats import sem

for n_corex_layer_to_use in range(len(corex_bundle['corex_layers'])+1):
    print('testing corex layers ', corex_bundle['dims'][:n_corex_layer_to_use])

    def corex_transform(dat):
        out = dat
        for layer in corex_bundle['corex_layers'][:n_corex_layer_to_use]:
            out = layer.transform(out)
        return out

    inspired_knnlm_recommender = KNNLMForTuningHyperParams(
        data_path= 'datasets/inspired_train.csv',
        heldout_portion=0.2
    )

    k=20
    n_neighbors = [1, 5, 20, 60, 180, 360, 720, 1024]
    recall = []

    test_query_embeddings = build_datastore_embedding(inspired_knnlm_recommender.validation_dataset['context']).cuda()

    for num_posts_to_consider in n_neighbors:
        hits_at_k = []
        # cache = dict()
        for idx in [i for i in range(test_query_embeddings.shape[0])]:
            query = test_query_embeddings[idx]
            target = inspired_knnlm_recommender.movie_vocab[inspired_knnlm_recommender.validation_dataset['label'][idx]]
            #if query not in cache:
            movie_counts = inspired_knnlm_recommender.count_based_probability(
                query, 
                num_posts_to_consider=num_posts_to_consider, 
                return_logits=True
            )
            recommended_movies = np.array(inspired_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
            # else:
            #     recommended_movies = cache[query]
            
            hits_at_k.append(int(target in recommended_movies[:k]))
        recall.append(np.mean(hits_at_k))


    best_n_neighbors = n_neighbors[np.argmax(recall)]
    print('the recommended number of neighbors to use is ', best_n_neighbors)

    testset = pd.read_csv('datasets/inspired_test.csv')
    test_inputs = testset['test_inputs']
    test_groundtruths = testset['test_outputs']

    inspired_knnlm_recommender = KNNLMForTuningHyperParams(
        data_path= 'datasets/inspired_train.csv',
        heldout_portion = 0.0001
    )

    test_query_embeddings = build_datastore_embedding(test_inputs).cuda()

    K = [1,5, 10, 20,50,100,300]

    print('-------------------- Retrieval -------------------------------')

    cache = dict()
    for k in K:
        hits_at_k = []
        for idx in range(len(test_inputs)):
            query = test_inputs[idx]
            target = test_groundtruths[idx]
        
            if query not in cache:
                movie_counts = inspired_knnlm_recommender.count_based_probability(
                    test_query_embeddings[idx].cuda(), 
                    num_posts_to_consider=  best_n_neighbors, 
                    return_logits=True
                )
                recommended_movies = np.array(inspired_knnlm_recommender.movie_vocab)[np.argsort(-movie_counts)]
                cache[query] = recommended_movies
            else:
                recommended_movies = cache[query]
            hits_at_k.append(int(target in recommended_movies[:k]))
        print('r@'+str(k), np.mean(hits_at_k),'; se: ', sem(hits_at_k))