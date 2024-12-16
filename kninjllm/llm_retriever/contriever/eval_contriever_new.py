import argparse
from functools import partial
from mteb import MTEB
import torch
from models_new import get_model
import json

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from root_config import RootConfig

def get_args():
    parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark')
    parser.add_argument('--output_dir', default='', type=str, metavar='N', help='output directory')
    parser.add_argument('--model_path', default='tmp-outputs/',type=str, metavar='N', help='which model to use')
    parser.add_argument('--sentence_pooling_method', default='cls', type=str, metavar='N', help='which model to use')
    parser.add_argument('--dtype', default='fp16', type=str, help='which model to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument('--max_passage_length', default=128, type=int)
    parser.add_argument('--add_instruction', default=False, action='store_true')
    parser.add_argument('--llm', default=False, action='store_true')
    parser.add_argument('--T5', default=False, action='store_true')
    parser.add_argument('--tasks_name', nargs='+', type=str)
    args = parser.parse_args()
    return args

def eval(args):
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print(f"args: \n{args}")
    
    tasks_name = args.tasks_name
    model = get_model(args)    
    print("Running: ", tasks_name)
    
    evaluation = MTEB(
        tasks=tasks_name,
        task_langs=['en']
    )

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.description['name']
        task_type: str = task_cls.description['type']

        if task_type == 'Classification':
            print('Set normalize to False for classification task')
            model.normalize = False
        else:
            print('Set normalize to {}'.format(model.normalize))
            model.normalize = True
        
        eval_splits = ["test" if task_name not in ['MSMARCO'] else 'validation']  
        evaluation = MTEB(tasks=[task_name], task_langs=['en']) 
        evaluation.run(
            model,
            output_folder=args.output_dir,
            eval_splits=eval_splits, 
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    
    
    args = get_args()
    
    args.model_path = RootConfig.contriever_model_path
    args.dtype = "fp16"
    args.sentence_pooling_method = "mean"
    
    
    args.tasks_name = "FiQA2018"
    
    args.output_dir = "./output"
    
    eval(args)
