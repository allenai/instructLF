import pandas as pd
from itertools import chain
from collections import defaultdict
from collections import Counter
import numpy as np
from tqdm import tqdm

def get_inspired_data(csv_path='datasets/inspired/inspired_large_train.csv', max_movies=20000):
    """
    Misc preprocessing for loading the dataset into training format.

    Returns:
        inspired_data: context and labels from the training split
        movie_vocab: most frequent N=max_movies movies
    """

    inspired_posts_train = pd.read_csv(csv_path)
    eligible_movies = [eval(i) for i in inspired_posts_train['movies']]
    frequency = Counter(list(chain.from_iterable(eligible_movies)))
    most_freq_N = max_movies
    print(f'interaction preservance rate at {most_freq_N} items')
    print(sum(i[1] for i in frequency.most_common(most_freq_N)) / sum(i[1] for i in frequency.items()))

    unique_movie_names = sorted(i[0] for i in frequency.most_common(most_freq_N))
    movie2id = {movie: idx for idx, movie in enumerate(unique_movie_names)}
    id2movie = {idx:movie for movie,idx in movie2id.items()}
    movie_vocab = unique_movie_names

    print('num items ', len(movie2id))
    positive_samples = []
    for post in eligible_movies:
        filtered_post = [movie for movie in post if movie in movie2id]
        positive_samples.append([movie2id[movie] for movie in filtered_post])

    inspired_data = {
        'context':[],
        'label':[]
    }

    # full situation is the entire post (there is another field that has post title only
    contexts = list(inspired_posts_train['full_situation'])
    print('flattening posts into training data')

    for i in tqdm(range(len(positive_samples)), total = len(positive_samples)):
        context = contexts[i]

        for positive_sample in positive_samples[i]:
            inspired_data['context'].append(context)
            inspired_data['label'].append(positive_sample)

    return inspired_data, movie_vocab


training_data, movie_vocab = get_inspired_data('./datasets/inspired_train.csv')

non_dup_contexts = sorted(list(set(training_data['context'])))

label_analysis_prompt = """help me analyze the following dialogue between the user and a movie recommender assistant. We are particularly interested in factors that affects what movie should the assistant discuss/recommend in the next response. Pay special attention to the task the current topic and the users' expressed interest in movie. 

Here is the interaction log:
<request> 

Generate a one sentence description of the dialogue's current state w.r.t. what type of movie to recommend next. what's the users' preference? are any properties of the next movie to discuss known to us?"""


from tqdm import tqdm
import openai
import numpy as np
import os
os.environ['OPENAI_API_KEY'] = None # put your openai key here

model_outputs = []
costs = 0

model_inputs = []


for request in tqdm(non_dup_contexts):

   
    input_prompt = label_analysis_prompt.replace('<request>', request)
    model_inputs.append(input_prompt)

import datasets
model_input_dataset = datasets.Dataset.from_dict({'model_inputs':model_inputs, 'queres':non_dup_contexts})


from llm_engines import get_call_worker_func
call_worker_func = get_call_worker_func(
    model_name="gpt-3.5-turbo-0125", 
    num_workers=16, # number of workers
    num_gpu_per_worker=0, # tensor parallelism size for each worker
    engine="openai", # or one of "vllm", "together", "openai", "mistral", "claude",
    use_cache=True
)


def map_generate(example):
    example['dialogue_analysis'] = call_worker_func([example['model_inputs']], temperature=0.0, max_tokens=None)
    return example

model_output_dataset_firststep = model_input_dataset.map(map_generate, num_proc=16) 


action_analysis_prompt = """Now, given your inferred current dialogue situation, propose a numbered list of property keywords that the next movie being discussed likely satisfy. E.g., "Romantic Genre", "Comedy Genre", "Features actor X", "Superhero movie", "Dark humor elements", etc. Your numbered list of properties:"""


def map_generate(example):
        example['annotations'] = call_worker_func([example['model_inputs'], example['dialogue_analysis'], action_analysis_prompt], temperature=0.0, max_tokens=None)
        return example

model_output_dataset_secondstep = model_output_dataset_firststep.map(map_generate, num_proc=16) 

import os
os.makedirs('./assets', exist_ok=True)
model_output_dataset_secondstep.to_csv('./assets/inspired_movie_annotations.csv')