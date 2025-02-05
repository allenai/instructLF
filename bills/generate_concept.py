def generate_concept():
    import joblib
    import os 
    import json 

    data_dir = 'data/topic_generation.jsonl'
    data = []

    with open(data_dir, 'r') as file:
        for line in file.readlines():
            data.append(json.loads(line))


    # data = data

    label_analysis_prompt = """help me analyze the following bill summary from U.S. congresses. We are particularly interested in factors that governs the topic this document addresses, e.g. trades, foreign trades, agriculture, etc.

    Here is the document:
    <document> 

    Generate a one sentence description of the key topics/directions addressed by this document."""


    model_inputs_label_analysis = []
    summaries = []
    for document in data:
        if len(document['summary'].split()) > 1000:
            print('truncating to avoid dead loop')
            document['summary'] = ' '.join(document['summary'].split()[:1000])
        model_inputs_label_analysis.append(label_analysis_prompt.replace('<document>', document['summary']))
        summaries.append(document['summary'])

    import os
    os.environ['OPENAI_API_KEY'] = None # put your openai key here

    import datasets
    model_input_dataset = datasets.Dataset.from_dict({'model_inputs':model_inputs_label_analysis, 'texts':summaries})

    from llm_engines import get_call_worker_func
    call_worker_func = get_call_worker_func(
        model_name="gpt-3.5-turbo-0125", 
        num_workers=16, # number of workers
        num_gpu_per_worker=0, # tensor parallelism size for each worker
        engine="openai", # or one of "vllm", "together", "openai", "mistral", "claude",
        use_cache=True
    )


    def map_generate(example):
        example['label_analysis'] = call_worker_func([example['model_inputs']], temperature=0.0, max_tokens=None)
        return example

    model_output_dataset_firststep = model_input_dataset.map(map_generate, num_proc=16) 

    action_analysis_prompt = """Now, based on your summary, propose a couple of potential high-level "topic" that describes what area of legislation this bill covers. E.g. "Economy", "Education", "Foreign Tax", "Gender Equality", etc. Your numbered list of topics:"""



    from tqdm import tqdm
    import openai
    import numpy as np
    import pandas as pd

    def map_generate(example):
        example['concepts'] = call_worker_func([example['model_inputs'], example['label_analysis'], action_analysis_prompt], temperature=0.0, max_tokens=None)
        return example

    model_output_dataset_secondstep = model_output_dataset_firststep.map(map_generate, num_proc=16) 


    import os
    os.makedirs('./assets', exist_ok=True)
    pd.DataFrame(model_output_dataset_secondstep).to_csv('./assets/bills_annotations.csv')


if __name__ == '__main__':
    generate_concept()



