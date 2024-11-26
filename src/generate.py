import os
import time 
import json 
import pprint 
from tqdm import tqdm
import pickle as pkl 
import pandas as pd 
import pdb 

import openai
from datasets import load_dataset, load_from_disk
import torch

from configs.config_generate import parse_args
from utils.utils import get_util_functions_self_cluster
from utils.trr_functions import get_pseudocode1, get_pseudocode2, get_trr, get_problem
  
    
args = parse_args()
argsdict = vars(args)
# print(pprint.pformat(argsdict))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
model_mapping = {"gpt3.5": "gpt-3.5-turbo-16k", 
                "gpt4": "gpt-4"}

API_KEY_FILE = 'openaiapikey.txt'
with open(API_KEY_FILE, 'r', encoding='utf-8') as infile:
    openai.api_key = infile.read()
if not openai.api_key:
    print("Could not find API key value at {}".format(API_KEY_FILE))
    sys.exit(1)

if args.split is not None and args.split == 'mini_val':
    apps = load_from_disk(f'data/{args.split}')
    apps_problem_ls = []
    for level in ['introductory', 'interview', 'competition']:
        apps_problem_ls += list(apps[level])
else:
    apps = load_dataset("codeparrot/apps")[args.split]

prelim_prompts = []
for prompt_i in range(len(args.prompt_file)):
    with open(args.prompt_file[prompt_i], 'r', encoding='utf-8') as infile:
        prelim_prompts.append(infile.read())

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, exist_ok=True)

if args.modules_file is not None: 
    if '.csv' in args.modules_file:
        modules = pd.read_csv(args.modules_file)
    else:
        modules = pkl.load(open(args.modules_file, 'rb'))

    modules = get_util_functions_self_cluster(modules, num_clusters=args.num_clusters)
    print("Util functions for {} problems".format(len(modules)))

get_problem = {
    "name": "get_problem",
    "description": "Get coding problem",
    "parameters": {
    },
}

get_trr = {
    "name": "get_trr",
    "description": "Get trr solving framework",
    "parameters": {
    },
}

get_pseudocode1 = {
    "name": "get_pseudocode1",
    "description": "Get Pseudocode",
    "parameters": {},
}

get_pseudocode2 = {
    "name": "get_pseudocode2",
    "description": "Get Pseudocode",
    "parameters": {
    },
}

lens = [] 
for idx in tqdm(range(args.start, args.end), total=args.end-args.start): 
    if args.split is not None and args.split == 'mini_val':
        problem = apps_problem_ls[idx]
    else:
        problem = apps[idx]
    problem_id = problem['problem_id']
    
    # if os.path.exists(args.output_path + '/{}.json'.format(problem_id)):
    #     continue 
            
    question = problem['question']
    if len(prelim_prompts) == 0:
        curr_prompt = prompt.replace("<<problem>>", question)
    else:
        # prelim_prompts[0] = prelim_prompts[0].replace("<<problem>>", question)
        # prelim_prompts[1] = prelim_prompts[1].replace("<<problem>>", question)
        # prelim_prompts[2] = prelim_prompts[2].replace("<<problem>>", question)
        with open('coding_problem.txt', 'w') as f:
            f.write(question)
        curr_prompt = prelim_prompts[-1]
    
    success = False 
    start = time.time()
    responses = [] 
    if args.num_gen_samples==1:
        num_loops = 1 
    else:
        num_loops = int(args.num_gen_samples/5)
    
    for i in tqdm(range(num_loops), leave=False):
        for stage in range(len(prelim_prompts)):
            prompt = prelim_prompts[stage]
            prompt = [{"role": "user", "content": prompt}]
            if stage == 0:
                response = openai.ChatCompletion.create(
                    model=model_mapping[args.model], 
                    messages=prompt,
                    n=1,
                    functions=[get_problem, get_trr],
                    function_call="auto"
                )
            elif stage == 1:
                response = openai.ChatCompletion.create(
                    model=model_mapping[args.model], 
                    messages=prompt,
                    n=1,
                    functions=[get_problem, get_pseudocode1],
                    function_call="auto"
                )
            elif stage == 2:
                response = openai.ChatCompletion.create(
                    model=model_mapping[args.model], 
                    messages=prompt,
                    n=1,
                    functions=[get_problem, get_pseudocode2],
                    function_call="auto"
                )

            if stage > 0:
                with open(f'pseudocode{stage}.txt', 'w') as f:
                    f.write(response.choices[0].message['content'])
            print(prompt, '\n\n\n\n', response)
            print('\n\n\n\n')
            
    if not response: 
        print("Failure to generate! skipp this sample problem id {}".format(problem_id))
        continue 
        
    if args.num_gen_samples == 1:
        result = response.choices[0].message['content']
    else:
        result = []
        for response in responses:
            result += [choice.message['content'] for choice in response.choices]
    
    curr_output = {} 
    curr_output['prompt'] = curr_prompt 
    curr_output['output'] = result 
    json.dump(curr_output, open(args.output_path + '/{}.json'.format(problem_id), 'w'))
