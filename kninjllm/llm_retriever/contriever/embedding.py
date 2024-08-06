import torch, faiss
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Any, List, Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoModel
import pytrec_eval, copy
import os, json, time, csv
from pprint import pprint

from root_config import RootConfig
from kninjllm.llm_retriever.BGE.model_eval import get_model
logger = logging.getLogger(__name__)

@dataclass
class Args:
    
    model_name: str = field(
        default="Bert-base",
        metadata={'help': 'The encoder name or path.'}
    )
    model_path: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={'help': 'The encoder name or path.'}
    )
    dtype: str = field(
        default='fp16',
        metadata={'help': 'Use fp16 in inference?'}
    )
    sentence_pooling_method: str = field(
        default='mean',
        metadata={'help': 'Use fp16 in inference?'}
    )
    normalized: bool = field(
        default=False,
        metadata={'help': 'Normlized'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    llm: bool = field(
        default=False,
        metadata={'help': 'Is the model LLM series?'}
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
        default=64,
        metadata={'help': 'Inference batch size.'}
    )
    
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=3,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    encoded_embedding_pth: str = field(
        default=RootConfig.root_path + "kninjllm/llm_retriever/BGE/FaissSave/Bert-base/fiqa_content.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )
    
    input_query: str = field(
        default=RootConfig.root_path + "kninjllm/llm_retriever/EvalDataset/fiqa/queries.jsonl",
        metadata={'help': 'dataset query.'}
    )
    corpus_pth: str = field(
        default=RootConfig.root_path + "kninjllm/llm_retriever/EvalDataset/fiqa/corpus.jsonl",
        metadata={'help': 'dataset corpus.'}
    )
    relation_pth: str = field(
        default=RootConfig.root_path + "kninjllm/llm_retriever/EvalDataset/fiqa/qrels/dev.jsonl",
        metadata={'help': 'relation between corpus and query.'}
    )
    results_dir: str = field(
        default=RootConfig.root_path + "kninjllm/llm_retriever/BGE/TestResult/Bert-base",
        metadata={'help': 'relation between corpus and query.'}
    )
    corpus_name: str = field(
        default="fiqa",
        metadata={'help': 'relation between corpus and query.'}
    )



def index(model, corpus, batch_size: int = 64, max_length: int = 256, index_factory: str = "Flat"):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    """
    corpus_embeddings = model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
    dim = corpus_embeddings.shape[-1]
    print(f"corpus_embeddings: {corpus_embeddings[:3]}")
    
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    
    co = faiss.GpuMultipleClonerOptions()
    
    co.shard = True
    faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
    

    
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(model, dtype, input_query, faiss_index: faiss.Index, k:int = 3):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    
    
    
    
    
    
    query_list = input_query

    
    query_embeddings = model.encode_queries(query_list, batch_size=1, max_length=256)
    all_scores = []
    all_indices = []
    for embed in tqdm(query_embeddings, desc="Searching"):
        score, indice = faiss_index.search(np.array([embed]).astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices


def load_corpus(corpus_pth):
    data = []
    with open(corpus_pth, 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)

    corpus_list, index_list, title_list = [], [], []
    for idx, item in enumerate(data[1:]):
        index_list.append(item[0])
        corpus_list.append(item[1])
        title_list.append(item[2])
    print(f"corpus_list example: {corpus_list[:2]}\nindex_list example: {index_list[:2]}\ntitle_list example: {title_list[:2]}")

    return corpus_list, title_list, index_list


def load_index(model, emb, index_factory):
    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(emb.shape[1], index_factory, metric)

    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, co)
    
    index.train(emb)
    index.add(emb)
    return index



def embedding(
    passages, 
    model_name_or_path,
    ):
    
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    args.model_path = model_name_or_path
    
    model = get_model(args=args, device=0)
    
    allids, allembeddings = model.encode_corpus_plus(passages, batch_size=1, max_length=256)
    print(f"allids[0]: {allids[0]}")
    print(f"allembeddings[0]: {allembeddings[0]}")
    
    return allids, allembeddings