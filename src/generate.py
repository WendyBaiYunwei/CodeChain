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

lens = [] 
for idx in tqdm(range(args.start, args.end), total=args.end-args.start): 
    if args.split is not None and args.split == 'mini_val':
        problem = apps_problem_ls[idx]
    else:
        problem = apps[idx]
    problem_id = problem['problem_id']
    
    if os.path.exists(args.output_path + '/{}.json'.format(problem_id)):
        continue 
            
    question = problem['question']
    if len(prelim_prompts) == 0:
        curr_prompt = prompt.replace("<<problem>>", question)
    else:
        prelim_prompts[1] = prelim_prompts[1].replace("<<problem>>", question)
        curr_prompt = prelim_prompts[-1]
    
    success = False 
    start = time.time()
    responses = [] 
    if args.num_gen_samples==1:
        num_loops = 1 
    else:
        num_loops = int(args.num_gen_samples/5)
    
    for i in tqdm(range(num_loops), leave=False):
        conversation = [{"role": "system", 
                        "content": "You are a helpful AI assistant to help developers to solve challenging coding problems."}]
        while time.time() - start < 80:   
            # preliminary question asking  
            for prelim_prompt in prelim_prompts[:-1]:
                prompt = {"role": "user", "content": prelim_prompt}
                conversation.append(prompt)
                response = openai.ChatCompletion.create(
                    model=model_mapping[args.model], 
                    messages=conversation,
                    n=1,
                )
                response = response.choices[0].message['content']
                response = {"role": "system", "content": response}
                conversation.append(response)
                    
            prompt = {"role": "user", 
                    "content": curr_prompt}   
            conversation.append(prompt)
            try: 
                response = openai.ChatCompletion.create(
                  model=model_mapping[args.model], 
                  messages=conversation,
                  n=1,
                )
                success = True 
                responses.append(response)
                break 
            except openai.error.InvalidRequestError as e:
                print(
                    f"Invalid request error: {e}"
                )
                time.sleep(60)
                continue 

            except openai.error.RateLimitError as e:
                print(
                    f"Rate limit error: {e}"
                )
                time.sleep(60)
                continue 

            except openai.error.APIError as e: 
                print(
                    f"HTTP code 502 API error: {e}"
                )
                time.sleep(60)
                continue 
        
    if not success: 
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