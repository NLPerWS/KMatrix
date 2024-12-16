




import pandas as pd
import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path
import hashlib
import random
import numpy as np
import torch
import transformers

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from kninjllm.llm_retriever.contriever.generate_passage_embeddings import main_do_embedding as do_contriever_embedding

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)
        print("type(ids) ",type(ids))
        print("type(embeddings) ",type(embeddings))
        
        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits
    pre_str = "Validation results: top k documents hits %s", top_k_hits
    print(pre_str)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return {"data":match_stats.questions_doc_hits,"message":message}


def add_passages(data, passages, top_passages_and_scores):
    
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "content": docs[c]["content"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main_bak(args):
    src.slurm.init_distributed_mode(args)

    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    
    passages = src.data.load_passages(args.passages)

    passage_id_map = {x["id"]: x for x in passages}

    
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    
    if args.dataSetList != [] and len(args.dataSetList) > 0 :
        data = args.dataSetList
        output_path = args.output_file_path
        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map, top_ids_and_scores)
        hasanswerObj = validate(data, args.validation_workers)
        hasanswer = hasanswerObj['data']
        message = hasanswerObj['message']
        
        add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
         
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")
        
    
    else:
        data_paths = glob.glob(args.data)
        alldata = []
        for path in data_paths:
            data = load_data(path)
            output_path = os.path.join(args.output_dir, os.path.basename(path))

            queries = [ex["question"] for ex in data]
            questions_embedding = embed_queries(args, queries, model, tokenizer)

            
            start_time_retrieval = time.time()
            top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
            add_passages(data, passage_id_map, top_ids_and_scores)
            hasanswerObj = validate(data, args.validation_workers)
            hasanswer = hasanswerObj['data']
            message = hasanswerObj['message']
            add_hasanswer(data, hasanswer)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as fout:
                for ex in data:
                    json.dump(ex, fout, ensure_ascii=False)
                    fout.write("\n")
            print(f"Saved results to {output_path}")

    return {"data":data,"message":message}



def main_evaluate_bak(args):
    
    src.slurm.init_distributed_mode(args)
    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()
    
    
    data = args.dataSetList
    queries = [ex["question"] for ex in data]
    print("---------------------len queries-------------------------")
    print(len(queries))
    
    final_top_ids_and_scores_config = {}
    hit_passage_id_map = {}
    
    for i,query in enumerate(queries):
        final_top_ids_and_scores_config[str(i)] = {
            "doc_id_list":[],
            "doc_score_list":[]
        }
    
    for fileName in os.listdir(args.passages_path):
        filePath = os.path.join(args.passages_path,fileName)
        passages = []
        with open(filePath, 'rb') as file:
            try:
                while True:
                    dict_obj = pickle.load(file)
                    dict_obj['title'] = dict_obj['content'].split("\t")[2]
                    passages.append(dict_obj)
            except EOFError:
                pass
        
        index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
        allids = []
        allembeddings = []
        for i, doc in enumerate(passages):
            allids.append(doc['id'])
            embeddings = doc['contriever_embedding']
            allembeddings.append(embeddings)
            if len(allembeddings) >= args.indexing_batch_size:
                allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
                allembeddings = []
        if allembeddings:
            allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
        print("Data indexing completed.")
        
        
        questions_embedding = embed_queries(args, queries, model, tokenizer)
        
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        
        
        
        
        for i,one_ids_and_scores in enumerate(top_ids_and_scores):

            filter_passages = list(filter(lambda x:x["id"] in one_ids_and_scores[0],passages))
            
            
            temp_passage_id_map = {x["id"]: x for x in filter_passages}
            
            
            
            hit_passage_id_map.update(temp_passage_id_map)
            
            final_top_ids_and_scores_config[str(i)]['doc_id_list'].extend(one_ids_and_scores[0])
            final_top_ids_and_scores_config[str(i)]['doc_score_list'].extend(one_ids_and_scores[1])

    final_top_ids_and_scores = []
    for con in final_top_ids_and_scores_config:
        temp_score_list = final_top_ids_and_scores_config[con]['doc_score_list']
        temp_id_list = final_top_ids_and_scores_config[con]['doc_id_list']
        
        
        
        
        top_indices = np.argsort(temp_score_list)[-args.n_docs:][::-1].tolist()
        
        
        
        
        new_score_list = []
        new_id_list = []
        for i in top_indices:
            new_score_list.append(temp_score_list[i])
            new_id_list.append(temp_id_list[i])

        final_top_ids_and_scores.append((new_id_list,np.array(new_score_list)))
        
        
    
    

    add_passages(data, hit_passage_id_map, final_top_ids_and_scores)
    hasanswerObj = validate(data, args.validation_workers)
    hasanswer = hasanswerObj['data']
    message = hasanswerObj['message']
    
    add_hasanswer(data, hasanswer)

    return {"data":data,"message":message}


def main_evaluate(args):
    
    data = main_infer(args)
    
    final_eva_data_list = []
    for origin_data,retriever_list in zip(args.dataSetList,data):
        final_eva_data_list.append({
            "answers":origin_data['golden_answers'],
            "ctxs":retriever_list,
            **origin_data
        })
    
    hasanswerObj = validate(final_eva_data_list, args.validation_workers)
    
    hasanswer = hasanswerObj['data']
    message = hasanswerObj['message']
    
    add_hasanswer(final_eva_data_list, hasanswer)
    
    with open(args.output_file_path,'w',encoding='utf-8') as f:
        json.dump(final_eva_data_list,f,ensure_ascii=False)
    
    return final_eva_data_list,message

def calculate_hash(s):
    sha256_hash = hashlib.sha256(s.encode()).hexdigest()
    return sha256_hash
    
def main_infer(args):
    
    temp_path = "temp_retriever_"+calculate_hash(str(time.time()) + str(random.randint(1, 1000000))) + ".jsonl"
    src.slurm.init_distributed_mode(args)
    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    if len(args.passages) > 0:
        passages = args.passages
        passages = [{"title":p['content'].split("\t")[2],**p} for p in passages]
        data = args.dataSetList
        queries = [ex["question"] for ex in args.dataSetList]
        hit_passage_id_map = {}
        
        questions_embedding = embed_queries(args, queries, model, tokenizer)
        print("---------------------len queries-------------------------")
        print(len(queries))
        
        
        index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
        
        allids = []
        allembeddings = []
        for i, doc in enumerate(passages):
            if doc['contriever_embedding'] == None:
                t_allids, t_allembeddings = do_contriever_embedding(passages=[doc])
                temp_id = doc['id']
                temp_embeddings = t_allembeddings[0].tolist()
            else:
                temp_id = doc['id']
                temp_embeddings = doc['contriever_embedding']
            allids.append(temp_id)
            allembeddings.append(temp_embeddings)
            if len(allembeddings) >= args.indexing_batch_size:
                allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
                allembeddings = []
        if allembeddings:
            allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
        print("Data indexing completed.")
    
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        
        passages_dict = {x["id"]: x for x in passages}
        for i, one_ids_and_scores in enumerate(top_ids_and_scores):
            filter_passages = [passages_dict[passage_id] for passage_id in one_ids_and_scores[0] if passage_id in passages_dict]
            temp_passage_id_map = {x["id"]: x for x in filter_passages}
            hit_passage_id_map.update(temp_passage_id_map)
        add_passages(data, hit_passage_id_map, top_ids_and_scores)    
        
        data = list(map(lambda x:x['ctxs'],data))
        
        return data
    

    elif args.passages_path != "":
        
        ok_count = 0
        
        queries = [ex["question"] for ex in args.dataSetList]
        
        questions_embedding = embed_queries(args, queries, model, tokenizer)
        print("---------------------len queries-------------------------")
        print(len(queries))
        dir_list = os.listdir(args.passages_path)
        for fileName in dir_list:
            
            data = args.dataSetList
            hit_passage_id_map = {}
            filePath = os.path.join(args.passages_path,fileName)
            
            passages = []
            with open(filePath, 'rb') as file:
                try:
                    while True:
                        dict_obj = pickle.load(file)
                        dict_obj['title'] = dict_obj['content'].split("\t")[2]
                        passages.append(dict_obj)
                        del dict_obj
                except EOFError:
                    pass
            
            
            index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

            allids = []
            allembeddings = []
            for i, doc in enumerate(passages):
                if doc['contriever_embedding'] == None:
                    t_allids, t_allembeddings = do_contriever_embedding(passages=[doc])
                    temp_id = doc['id']
                    temp_embeddings = t_allembeddings[0].tolist()
                else:
                    temp_id = doc['id']
                    temp_embeddings = doc['contriever_embedding']
                allids.append(temp_id)
                allembeddings.append(temp_embeddings)
                if len(allembeddings) >= args.indexing_batch_size:
                    allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
                    allembeddings = []
            if allembeddings:
                allembeddings, allids = add_embeddings(index, np.array(allembeddings), allids, args.indexing_batch_size)
            print("Data indexing completed.")

            
            start_time_retrieval = time.time()
            top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
            
            
            
            
            passages_dict = {x["id"]: x for x in passages}
            for i, one_ids_and_scores in enumerate(top_ids_and_scores):
                filter_passages = [passages_dict[passage_id] for passage_id in one_ids_and_scores[0] if passage_id in passages_dict]
                temp_passage_id_map = {x["id"]: x for x in filter_passages}
                hit_passage_id_map.update(temp_passage_id_map)
                
            add_passages(data, hit_passage_id_map, top_ids_and_scores)
            
            
            
            with open(temp_path, 'a',encoding='utf-8') as file:
                file.write(json.dumps(data) + '\n')
            ok_count += 1
            
            
            del passages
            del data
            del allembeddings
            del allids
            del hit_passage_id_map
            del top_ids_and_scores
            del passages_dict
            del filter_passages
            del temp_passage_id_map
            del index
                    
        
        
        temp_data = []
        with open(temp_path, 'r',encoding='utf-8') as file:
            for d in file:
                 temp_data.append(json.loads(d))
        
        
        
        
        
        
        
        final_data_dict = {
            query['question']+str(index): {**query, 'ctxs': []}
            for index,query in enumerate(args.dataSetList)
        }
        
        
        
        for line in temp_data:
            for index,item in enumerate(line):

                if item['question']+str(index) in final_data_dict:
                    final_data_dict[item['question']+str(index)]['ctxs'].extend(item['ctxs'])

        
        for data_obj in final_data_dict.values():
            
            unique_data = {item['id']: item for item in data_obj['ctxs']}.values()

            
            sorted_data = sorted(unique_data, key=lambda x: float(x['score']), reverse=True)
            data_obj['ctxs'] = sorted_data[0:args.n_docs]

        final_data_list = list(final_data_dict.values())

        final_data_list = list(map(lambda x:x['ctxs'],final_data_list))

        
        
        if os.path.exists(temp_path):
            os.remove(temp_path) 
        
            
        return final_data_list


    else:
        raise ValueError("The retriever has no knowledge source...")

def find_top_n_indices(arr, n):
    indices = []
    for _ in range(n):
        max_val = float('-inf')
        max_idx = None
        for i in range(len(arr)):
            if i in indices:
                continue
            if arr[i] > max_val:
                max_val = arr[i]
                max_idx = i
        indices.append(max_idx)
    return sorted(indices, key=lambda x: arr[x], reverse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    args = parser.parse_args()
    print("-----------------------------args-----------------------")
    print(args)
    print("--------------------------------------------------------")
    
    main(args)


