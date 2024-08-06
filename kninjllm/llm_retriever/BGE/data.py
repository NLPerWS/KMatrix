import math
import os.path
import random, time
from dataclasses import dataclass
import torch
import numpy as np
import datasets
from pprint import pprint
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import torch.distributed as dist

from arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            tokenizer,
            args: DataArguments,
    ):
        args.train_data = args.train_data[0]
        print(f"args.train_data: {args.train_data}")
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        self.model_type = args.model_type
        self.use_instruction = args.use_instruction
        self.print_flag = True

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):

        query = self.dataset[item]['query']
        if self.use_instruction:
            query = f"Instruct: {query_instruction_for_retrieval} \nQuery: {query}"
        
        passages = []
        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.print_flag:
            print(f"query: \n{query}")
            print(f"passages: \n{passages}")
            self.print_flag = False

        return query, passages


class SameDatasetTrainDataset(Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same task.
    """
    def __init__(self, tokenizer, args: DataArguments, batch_size: int, seed: int, process_index: int=0, num_processes: int=1):
        train_datasets = []
        each_data_inxs = []
        batch_size_inxs = []
        pqloss_flag = []
        cur_all_num = 0
        
        SMALL_THRESHOLD = args.small_threshold
        DROP_THRESHOLD = args.drop_threshold
        
        context_feat = datasets.Features({
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string')),
            'query_instruction': datasets.Value('string'),
            'passage_instruction': datasets.Value('string'),
            'pos_scores': datasets.Sequence(datasets.Value('float')),
            'neg_scores': datasets.Sequence(datasets.Value('float')),
            'fewshot_example_query': datasets.Sequence(datasets.Value('string')),
            'fewshot_example_passage': datasets.Sequence(datasets.Value('string')),
        })
        
        assert isinstance(args.train_data, list) and len(args.train_data) >= 1
        
        if dist.get_rank() == 0:
            self.print_batch_size(batch_size=batch_size, train_group_size=args.train_group_size)
        
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"{data_dir} is a file, not a directionary")
            
            small_datasets = []
            small_batch_size = math.inf
            
            # Add `parallel_` in `data_dir` to indicate that this dataset is parallel corpus
            flag = 'parallel_' in data_dir
            for file in os.listdir(data_dir):
                if not (file.endswith('.json') or file.endswith('.jsonl')):
                    continue
                
                file_path = os.path.join(data_dir, file)
                if dist.get_rank() == 0:
                    print(f'loading data from {file_path} ...')

                temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat)
                if not args.knowledge_distillation:
                    try:
                        temp_dataset = temp_dataset.remove_columns(['pos_scores', 'neg_scores'])
                    except:
                        pass
                
                if len(temp_dataset) == 0:
                    continue
                elif len(temp_dataset) < SMALL_THRESHOLD:
                    small_datasets.append(temp_dataset)
                    small_batch_size = min(small_batch_size, self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                else:
                    if args.max_example_num_per_dataset is not None and len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
                    each_data_inxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                    cur_all_num += len(temp_dataset)
                    batch_size_inxs.append(self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                    pqloss_flag.append(flag)
            
            if len(small_datasets) > 0:
                small_dataset = datasets.concatenate_datasets(small_datasets)
                if len(small_dataset) >= DROP_THRESHOLD:
                    train_datasets.append(small_dataset)
                    each_data_inxs.append(np.arange(len(small_dataset)) + cur_all_num)
                    cur_all_num += len(small_dataset)
                    batch_size_inxs.append(small_batch_size)
                    pqloss_flag.append(flag)
        
        self.dataset = datasets.concatenate_datasets(train_datasets)
        self.each_data_inxs = each_data_inxs
        self.datasets_inxs = np.arange(len(each_data_inxs))
        self.batch_size_inxs = batch_size_inxs
        self.pqloss_flag = pqloss_flag
        
        self.process_index = process_index
        self.num_processes = num_processes
        self.args = args
        self.shuffle_ratio = args.shuffle_ratio
        
        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self.refresh_epoch()
        self.print_flag=True
        self.print_template=True
        self.cur_batch_indices = 0
        
        self.model_type = args.model_type
        self.use_instruction = args.use_instruction
        self.tokenizer = tokenizer
        
    
    def print_batch_size(self, batch_size: int, train_group_size: int):
        length_list = ['0-500', '500-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-inf']
        batch_size_dict = {
            k: self.get_file_batch_size(f"len-{k}.jsonl", batch_size, train_group_size) for k in length_list
        }
        batch_size_list = [
            f'{length}: {batch_size_dict[length]}' for length in length_list
        ]
        print("=========================")
        print("Batch Size Dict:")
        pprint(batch_size_list)
        print("=========================")
    
    @staticmethod
    def get_file_batch_size(file: str, batch_size: int, train_group_size: int):
        if train_group_size == 8:
            return batch_size
        elif train_group_size == 1:
            return batch_size
        else:
            return batch_size
    
    
    def refresh_epoch(self):
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')
        self.deterministic_generator.shuffle(self.datasets_inxs)
        # Dynamically adjust batch size
        batch_datas = []
        for dataset_inx in self.datasets_inxs:
            self.deterministic_generator.shuffle(self.each_data_inxs[dataset_inx])
            cur_batch_size = self.batch_size_inxs[dataset_inx]*self.num_processes
            flag = self.pqloss_flag[dataset_inx]
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                # judge the last batch's length
                if len(self.each_data_inxs[dataset_inx]) - start_index < 2 * self.num_processes:
                    break
                batch_datas.append((self.each_data_inxs[dataset_inx][start_index:start_index+cur_batch_size], flag))
        self.deterministic_generator.shuffle(batch_datas)
        self.batch_datas = batch_datas
        self.step = 0


    def __getitem__(self, _):  
        batch_indices, pqloss_flag = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.cur_batch_indices = batch_indices
        self.step += 1
        queries, passages, teacher_scores = self.create_batch_data(batch_raw_data=batch_data)
        # print('rank, step, flag, query, passage:', dist.get_rank(), self.step, pqloss_flag, queries, passages)
        return queries, passages, teacher_scores, pqloss_flag


    def shuffle_text(self, text):
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text


    def get_template(self, query_instruction=None, query=None, fewshot_example_query=None, fewshot_example_passage=None, passage=None):
        if self.use_instruction:
            template = f"Instruct: {query_instruction} \nQuery: {query}"
        else:
            template = f"{query}"
        if self.print_template:
            print(f"query: {template}")
            self.print_template=False
        return template
    
    
    def create_batch_data(self, batch_raw_data):
        
        queries, passages, query_insts, passage_insts = [], [], [], []
        teacher_scores = []
        for i in range(len(batch_raw_data['query'])):
            query_insts.append(batch_raw_data['query_instruction'][i])
            passage_insts.append(batch_raw_data['passage_instruction'][i])

            query_template = self.get_template(
                query_instruction=batch_raw_data['query_instruction'][i],
                query=batch_raw_data['query'][i],
                fewshot_example_query=batch_raw_data['fewshot_example_query'][i],
                fewshot_example_passage=batch_raw_data['fewshot_example_passage'][i],
                )
            queries.append(query_template)
            
            pos_inx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            passages.append(self.shuffle_text(batch_raw_data['pos'][i][pos_inx]))
            if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                teacher_scores.append(batch_raw_data['pos_scores'][i][pos_inx])
            
            neg_inx_set = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_inxs = random.sample(neg_inx_set * num, self.args.train_group_size - 1)
            else:
                neg_inxs = random.sample(neg_inx_set, self.args.train_group_size - 1)
                     
            negs = [batch_raw_data['neg'][i][x] for x in neg_inxs]
            passages.extend(negs)
            
            if len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)
        
        
        if self.print_flag==True:
            print(f"final query: {queries}")
            print(f"final passages: {passages}")
            self.print_flag=False
        
        if len(teacher_scores) == 0:
            teacher_scores = None
        return queries, passages, teacher_scores
    
    def __len__(self):
        return len(self.batch_datas) * self.num_processes


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(
            self,
            tokenizer,
            args: DataArguments,
            query_max_len: int = 64,
            passage_max_len: int = 256,
    ):
        self.print_flag = True
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.args = args


    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        
        teacher_scores = None
        if len(features[0]) > 2:
            teacher_scores = [f[2] for f in features]
            if teacher_scores[0] is None:
                teacher_scores = None
            else:
                teacher_scores = torch.FloatTensor(teacher_scores)
        
        flag = None
        if len(features[0]) == 4:
            flag = [f[3] for f in features][0]
            
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            #return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            #return_tensors="pt",
        )
        
        if self.args.model_type == "LLM":
            q_collated['input_ids'] = torch.tensor([ids + [self.tokenizer.eos_token_id] for ids in q_collated['input_ids']])
            q_collated['attention_mask'] = torch.tensor([mask + [1] for mask in q_collated['attention_mask']])
            d_collated['input_ids'] = torch.tensor([ids + [self.tokenizer.eos_token_id] for ids in d_collated['input_ids']])
            d_collated['attention_mask'] = torch.tensor([mask + [1] for mask in d_collated['attention_mask']])
        else:
            q_collated['input_ids'] = torch.tensor(q_collated['input_ids'])
            q_collated['attention_mask'] = torch.tensor(q_collated['attention_mask'])
            q_collated['token_type_ids'] = torch.tensor(q_collated['token_type_ids'])
            d_collated['input_ids'] = torch.tensor(d_collated['input_ids'])
            d_collated['attention_mask'] = torch.tensor(d_collated['attention_mask'])
            d_collated['token_type_ids'] = torch.tensor(d_collated['token_type_ids'])

        if self.print_flag:
            print(f"q_collated: {q_collated}")
            print(f"d_collated: {d_collated}")
            self.print_flag=False
        
        return {"query": q_collated, "passage": d_collated}
