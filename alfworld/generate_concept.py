def generate_concept():
    import joblib
    import os 

    data_dir = 'compile_data/compiled_sequences/train'
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

    # data = data

    label_analysis_prompt = """help me analyze the following action log ("input)" of a user. We are particularly interested in factors that affects what the user would do next. Pay special attention to the task the user is given and the users' newest state in the interaction log. 

    Here is the interaction log:
    <interaction_log> 

    Generate a one sentence description of the users' current process w.r.t. completing the given task - what still needs to be accomplished? what's already accomplished? Any nearby objects/utensils that matters for completing the task? what's the user's given task?"""


    model_inputs_label_analysis = []
    gold_actions = []
    states = []
    for sequence in data:
        traj = get_trajectory_text(sequence)
        teacher_forcing_data = get_teacher_forcing_data(sequence)
        for dp in teacher_forcing_data:
            info = {'whole_trajectory': get_trajectory_text(sequence), 'cur_state': dp['input'], 'gold_action':dp['label']}
            info = 'Input: '+str(info['cur_state'])
            model_inputs_label_analysis.append(label_analysis_prompt.replace('<interaction_log>', info))
            gold_actions.append(dp['label'][1])
            states.append(dp['input'])

    import os
    os.environ['OPENAI_API_KEY'] = None # put you openai key here

    import datasets
    model_input_dataset = datasets.Dataset.from_dict({'model_inputs':model_inputs_label_analysis, 'states':states, 'gold_actions':gold_actions})

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

    action_analysis_prompt = """Now, given your inferred current user situation, propose a numbered list of property keywords that describes the users' current status w.r.t. task completion, e.g. "already cleaned an item", "a microwave is at current location", "the user is tasked with cleaning an item", "the user needs to find a pot", etc. Your numbered list of properties:"""



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
    pd.DataFrame(model_output_dataset_secondstep).to_csv('./assets/alfworld_annotations.csv')

if __name__ == '__main__':
    generate_concept()

