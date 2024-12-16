import multiprocessing
import logging
import datasets
from dataclasses import dataclass, field
import datasets
import pandas as pd
from transformers import HfArgumentParser
from tqdm import tqdm
import torch, faiss
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
import pytrec_eval
import os, json, time
from root_config import RootConfig

from model_eval import get_model

logger = logging.getLogger(__name__)

@dataclass
class Args_encode():
    # model args
    model_name: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'The encoder name or path.'}
    )
    model_path: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'The encoder name or path.'}
    )
    dtype: str = field(
        default='bf16',
        metadata={'help': 'Use fp16 in inference?'}
    )
    sentence_pooling_method: str = field(
        default='eos',
        metadata={'help': 'Use fp16 in inference?'}
    )
    normalized: bool = field(
        default=True,
        metadata={'help': 'normlized'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    llm: bool = field(
        default=True,
        metadata={'help': 'Add query-side instruction?'}
    )
    T5: bool = field(
        default=False,
        metadata={'help': 'Is the model T5 series?'}
    )    
    max_query_length: int = field(
        default=64,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=256,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=16,
        metadata={'help': 'Inference batch size.'}
    )
    # retrieval args
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=1000,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    save_dir: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'Path to save embeddings.'}
    )
    # data args
    query_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/queries.jsonl",
        metadata={'help': 'dataset query.'}
    )
    corpus_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/corpus.jsonl",
        metadata={'help': 'dataset corpus.'}
    )
    relation_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/qrels/test.tsv",
        metadata={'help': 'relation between corpus and query.'}
    )
    # multi process arg
    num_device: int = field(
        default=1,
        metadata={'help': 'the number of process.'}
    )


@dataclass
class Args_eval:
    # model args
    model_name: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'The encoder name or path.'}
    )
    
    # retrieval args
    save_dir: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'Path to save embeddings.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=1000,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    # data args
    results_dir: str = field(
        default=RootConfig.E5_model_path,
        metadata={'help': 'relation between corpus and query.'}
    )
    eval_ds_name: str = field(
        default="ArguAna",
        metadata={'help': 'relation between corpus and query.'}
    )
    query_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/queries.jsonl",
        metadata={'help': 'dataset query.'}
    )
    corpus_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/corpus.jsonl",
        metadata={'help': 'dataset corpus.'}
    )
    relation_pth: str = field(
        default=RootConfig.root_path + "dir_dataset/retriever/test_data/ArguAna/qrels/test.tsv",
        metadata={'help': 'relation between corpus and query.'}
    )
    

def read_tsv(pth):
    relation = {}
    num_file = sum([1 for i in open(pth, "r")])
    with open(pth, 'r') as file:
        next(file) 
        for line in tqdm(file,total=num_file, desc="read tsv"):
            tmp_line = line.strip()
            columns = tmp_line.split('\t')[:2]
            q_id = str(columns[0])
            c_id = str(columns[1])
            if q_id not in relation:
                relation[q_id] = [c_id]
            else:
                relation[q_id].append(c_id)
    return relation
    
def read_jsonl(pth):
    data = []
    num_file = sum([1 for i in open(pth, "r")])
    with open(pth, 'r') as f:
        for line in tqdm(f,total=num_file,desc="read jsonl"):
            data.append(json.loads(line))
    return data

def get_data(args):
    eval_data = read_jsonl(args.query_pth)
    corpus = read_jsonl(args.corpus_pth)
    relation = read_tsv(args.relation_pth)  
    
    if 'title' in corpus[0]:
        corpus = [ {'_id':cops['_id'], 'text':cops['title'] + ' ' + cops['text']} for cops in corpus]
    
    corpus_dict = {}
    corpus_list = []
    content_2_id = {}
    for idx, cops in enumerate(corpus):
        corpus_dict[cops['_id']] = cops['text']
        corpus_list.append(cops['text'])
        content_2_id[cops['text']] = cops['_id']

    queries_dict = {}
    query_2_id = {}
    for qry in eval_data:
        queries_dict[qry['_id']] = qry['text']
        query_2_id[qry['text']] = qry['_id']
    
    ground_truth = []
    queries_list = []
    
    for item in relation.items():
        qry_id = item[0]
        corpus_ids = item[1]
        queries_list.append(queries_dict[qry_id])
        tmp_positives = [corpus_dict[cid] for cid in corpus_ids]
        ground_truth.append(tmp_positives)
    
    eval_data = {
    "query": queries_list,
    "positive": ground_truth
    }
    eval_data = datasets.Dataset.from_dict(eval_data)
    
    corpus = {
    "content": corpus_list,
    }
    corpus = datasets.Dataset.from_dict(corpus)
    
    return eval_data, corpus, content_2_id, query_2_id

def infer(model,
          corpus: datasets.Dataset,
          batch_size: int = 256,
          max_length: int=512,
          save_dir: str = None,
          device: int = 0,
          label: str = "content",
          num: int = 8,
          shared_hidden_size=None):
    """
    1. Encode the entire into dense embeddings;
    """
    # corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_length)
    if not isinstance(corpus, list):
        length = len(corpus[label])
        corpus = corpus[label][:]
    else:
        length = len(corpus)
    start = device * length // num
    if device == num - 1:
        end = length
    else:
        end = (device + 1) * length // num
    # start = 0
    # end = length
    if label == 'content':
        corpus_embeddings = model.encode_corpus(corpus[start:end], batch_size=batch_size, max_length=max_length)
    else:
        corpus_embeddings = model.encode_queries(corpus[start:end], batch_size=batch_size, max_length=max_length)
    
    #shared_hidden_size.value = corpus_embeddings.shape[-1]
    os.makedirs(save_dir, exist_ok=True)
    memmap_array = np.memmap(
        os.path.join(save_dir, f'{label}{device}.memmap'),
        shape=corpus_embeddings.shape,
        mode="w+",
        dtype=corpus_embeddings.dtype
    )
    memmap_array[:] = corpus_embeddings[:]

def worker_function(device, eval_data, corpus, shared_hidden_size,model_path,dataset_path,output_path):
    parser = HfArgumentParser([Args_encode])
    args_encode: Args_encode = parser.parse_args_into_dataclasses()[0]
    
    
    args_encode.model_path = model_path
    args_encode.query_pth = os.path.join(dataset_path, 'queries.jsonl')
    args_encode.corpus_pth = os.path.join(dataset_path, 'corpus.jsonl')
    args_encode.relation_pth = os.path.join(dataset_path, 'qrels', 'test.tsv')
    
    
    model = get_model(args_encode, device)

    queries = eval_data['query']
    infer(model=model,
        corpus=queries,
        batch_size=args_encode.batch_size,
        max_length=args_encode.max_query_length,
        save_dir=args_encode.save_dir,
        device=device,
        label='query',
        num=args_encode.num_device,
        shared_hidden_size=shared_hidden_size
        )

    infer(model=model,
          corpus=corpus,
          batch_size=args_encode.batch_size,
          max_length=args_encode.max_passage_length,
          save_dir=args_encode.save_dir,
          device=device,
          label='content',
          num=args_encode.num_device,
          shared_hidden_size=shared_hidden_size
          )
    model.cpu()

def merge(shared_hidden_size,model_path,dataset_path,output_path):
    parser = HfArgumentParser([Args_encode])
    args_encode: Args_encode = parser.parse_args_into_dataclasses()[0]

    args_encode.model_path = model_path
    args_encode.query_pth = os.path.join(dataset_path, 'queries.jsonl')
    args_encode.corpus_pth = os.path.join(dataset_path, 'corpus.jsonl')
    args_encode.relation_pth = os.path.join(dataset_path, 'qrels', 'test.tsv')
    
    hidden_size = shared_hidden_size
    print('hidden_size: ', hidden_size)

    merged_embeddings = np.empty((0, hidden_size), dtype=np.float32)

    num = args_encode.num_device

    for i in range(num):
        file_path = os.path.join(args_encode.save_dir, f'content{i}.memmap')
        embeddings_chunk = np.memmap(file_path, mode="r", dtype=np.float32).reshape(-1, hidden_size)
        merged_embeddings = np.vstack((merged_embeddings, embeddings_chunk))
        os.remove(file_path)

    corpus_array = np.memmap(
        os.path.join(args_encode.save_dir, f'content.memmap'),
        shape=merged_embeddings.shape,
        mode="w+",
        dtype=merged_embeddings.dtype
    )
    corpus_array[:] = merged_embeddings[:]

    merged_embeddings = np.empty((0, hidden_size), dtype=np.float32)

    for i in range(num):
        file_path = os.path.join(args_encode.save_dir, f'query{i}.memmap')
        embeddings_chunk = np.memmap(file_path, mode="r", dtype=np.float32).reshape(-1, hidden_size)
        merged_embeddings = np.vstack((merged_embeddings, embeddings_chunk))
        os.remove(file_path)

    query_array = np.memmap(
        os.path.join(args_encode.save_dir, f'query.memmap'),
        shape=merged_embeddings.shape,
        mode="w+",
        dtype=merged_embeddings.dtype
    )
    query_array[:] = merged_embeddings[:]


def main_encode(model_path,dataset_path,output_path):
    processes = []
    #multiprocessing.set_start_method('spawn')
    parser = HfArgumentParser([Args_encode])
    args_encode: Args_encode = parser.parse_args_into_dataclasses()[0]
    args_encode.model_path = model_path
    args_encode.query_pth = os.path.join(dataset_path, 'queries.jsonl')
    args_encode.corpus_pth = os.path.join(dataset_path, 'corpus.jsonl')
    args_encode.relation_pth = os.path.join(dataset_path, 'qrels', 'test.tsv')
    eval_data, corpus, content_2_id, query_2_id = get_data(args_encode)

    print(corpus[0])
    #shared_hidden_size = multiprocessing.Value('i', 0)
    shared_hidden_size = 768
    worker_function(0, eval_data, corpus, shared_hidden_size, model_path, dataset_path, output_path)
    # for i in range(args_encode.num_device):
    # # i = 7
    #     process = multiprocessing.Process(target=worker_function, args=(i, eval_data, corpus, shared_hidden_size,model_path,dataset_path,output_path))
    #     processes.append(process)
    #     process.start()

    # for process in processes:
    #     process.join()

    merge(shared_hidden_size, model_path, dataset_path, output_path)


def index(dtype, hidden_size, save_path: str = None, index_factory: str = "Flat"):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    print(f"save path: {save_path}")
    print(f"hidden_size: {hidden_size}")
    corpus_embeddings = np.memmap(
        save_path,
        mode="r",
        dtype=dtype
    ).reshape(-1, hidden_size)
    
    # create faiss index
    faiss_index = faiss.index_factory(hidden_size, index_factory, faiss.METRIC_INNER_PRODUCT)

    # co = faiss.GpuClonerOptions()
    co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    co.shard = True
    # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(dtype, hidden_size, save_path, faiss_index: faiss.Index, k:int = 100):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """

    query_embeddings = np.memmap(
        save_path,
        mode="r",
        dtype=dtype
    ).reshape(-1, hidden_size)

    query_size = len(query_embeddings)
    all_scores = []
    all_indices = []
    
        
    for embed in tqdm(query_embeddings, desc="Searching"):
        score, indice = faiss_index.search(np.array([embed]), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    

def mrr_recall(preds, labels, cutoffs=[1,10,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = round(mrr, 5)

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = round(recall, 5)

    return metrics

# update evaluate metric
def evaluate(queries, content_2_id, scores, retrieval_results, ground_truths, cutoffs=[1,3,5,10,100]):
    
    qrels = {}
    results = {}
    for query_id, item in tqdm(enumerate(queries), total=len(queries)):
        query = item['query']

        query_dict = {}

        for positive in ground_truths[query_id]:
            positive_id = content_2_id[positive]
            query_dict.update( {str(positive_id): 1})

        result_dict = {}

        for sc, pred in zip(scores[query_id], retrieval_results[query_id]):
            pred_id = content_2_id[pred]
            result_dict.update( {str(pred_id):float(sc)} )

        qrels[str(query_id)] = query_dict
        results[str(query_id)] = result_dict

    k_values = cutoffs
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    
    scores = evaluator.evaluate(results)
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    merged_dict = {k: v for d in [ndcg, _map, recall, precision] for k, v in d.items()}
    mrr_recall_metric = mrr_recall(retrieval_results, ground_truths, cutoffs=cutoffs)
    merged_dict.update(mrr_recall_metric)
    return merged_dict


def get_hidden_size(model_name):
    model_hidden_size = {
        'BGE': 768,
        'dpr': 768,
        'contriever': 768,
        'Bert-base': 768,
        'Bert-large': 1024,
        'Qwen1.5-0.5b': 1024,
        'Qwen1.5-4B': 2560,
        'Phi-v1.5-1.3b': 2048,
        'Phi-v2-2.7b': 2560,
        'Gemma-2b': 2048,
        'Gemma-7b': 3072,
        'Llama-2-7b': 4096,
        'repllama-v1-7b': 4096,
        'Llama-2-13b': 5120,
        'Qwen1.5-14B': 5120,
        'Qwen1.5-32B': 5120,
    }
    for name in model_hidden_size.keys():
        if name in model_name:
            return model_hidden_size[name]


def main_eval(model_path,dataset_path,output_path):
    parser = HfArgumentParser([Args_eval])
    args_eval: Args_eval = parser.parse_args_into_dataclasses()[0]

    args_eval.model_path = model_path
    args_eval.eval_ds_name = dataset_path.split("/")[-1]
    args_eval.results_dir = output_path
    args_eval.query_pth = os.path.join(dataset_path, 'queries.jsonl')
    args_eval.corpus_pth = os.path.join(dataset_path, 'corpus.jsonl')
    args_eval.relation_pth = os.path.join(dataset_path, 'qrels', 'test.tsv')
    
    print(f"args_eval: {args_eval}")
    eval_data, corpus, content_2_id, query_2_id = get_data(args_eval)
    
    results_dir = args_eval.results_dir
    print(f"Save Evaluation results to {results_dir}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directory '{results_dir}' created successfully.")
    result_path = os.path.join(results_dir, args_eval.eval_ds_name + '.json')
    if os.path.exists(result_path):
        print(f'skipped over {args_eval.model_name}')
        return
        
    dtype = 'float32'
    hidden_size = get_hidden_size(args_eval.model_name)

    copus_embed_save_path = os.path.join(args_eval.save_dir, "content.memmap")
    print('Loading corpus embedding...')
    faiss_index = index(
        dtype='float32',
        hidden_size=hidden_size,
        index_factory=args_eval.index_factory,
        save_path=copus_embed_save_path,
    )
    print('Loading query embedding, search...')
    query_embed_save_path = os.path.join(args_eval.save_dir, "query.memmap")
    scores, indices = search(
        dtype='float32',
        hidden_size=hidden_size,
        save_path=query_embed_save_path,
        faiss_index=faiss_index, 
        k=args_eval.k, 
    )
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["content"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])

    metrics = evaluate(eval_data, content_2_id, scores, retrieval_results, ground_truths, cutoffs=[1,3,5,10,100,1000])
    print(metrics)

    with open(result_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    return metrics


def test(model_path,dataset_path,output_path):
    main_encode(model_path,dataset_path,output_path)
    metrics = main_eval(model_path,dataset_path,output_path)    
    return metrics

# if __name__ == "__main__":
#     test(
#         model_path=RootConfig.E5_model_path,
#         dataset_path=RootConfig.root_path + "dir_dataset/retriever/test_data/NFCorpus",
#         output_path="./TestResult",
#     )