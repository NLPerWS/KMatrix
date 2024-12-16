import os
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import hashlib


from root_config import RootConfig
from kninjllm.llm_store.elasticsearch_document_store import ElasticsearchDocumentStore
from kninjllm.llm_retriever.contriever.src.contriever import *

from kninjllm.llm_retriever.contriever.generate_passage_embeddings import main_do_embedding as do_contriever_embedding
from kninjllm.llm_retriever.BGE.embedding import embedding as do_BGE_embedding

from kninjllm.llm_retriever.contriever.generate_passage_embeddings import main_do_embedding as do_DPR_embedding
from kninjllm.llm_retriever.contriever.generate_passage_embeddings import main_do_embedding as do_E5_embedding
from kninjllm.llm_retriever.contriever.generate_passage_embeddings import main_do_embedding as do_BERT_embedding


import torch
from transformers import AutoTokenizer   
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from vllm import RequestOutput, SamplingParams,LLM
from refined.inference.processor import Refined

def set_proxy():
    if RootConfig.HTTP_PROXY != "":
        os.environ['HTTP_PROXY']=RootConfig.HTTP_PROXY
    if RootConfig.HTTPS_PROXY != "":
        os.environ['HTTPS_PROXY']=RootConfig.HTTPS_PROXY
    

def unset_proxy():
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
        subprocess.call(['unset', 'HTTP_PROXY'], shell=True)
        
    if 'HTTPS_PROXY' in os.environ:
        del os.environ['HTTPS_PROXY']
        subprocess.call(['unset', 'HTTPS_PROXY'], shell=True)
    
    if 'HF_HOME' in os.environ:
        del os.environ['HF_HOME']
        subprocess.call(['unset', 'HF_HOME'], shell=True)
        
    if 'OPENBLAS_NUM_THREADS' in os.environ:
        del os.environ['OPENBLAS_NUM_THREADS']
        subprocess.call(['unset', 'OPENBLAS_NUM_THREADS'], shell=True)
        
    if 'HF_ENDPOINT' in os.environ:
        del os.environ['HF_ENDPOINT']
        subprocess.call(['unset', 'HF_ENDPOINT'], shell=True)
    
    
    
def _load_pickle_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def _reverse_dict(ori_dict):
    reversed_dict = {v: k for k, v in ori_dict.items()}
    return reversed_dict

def RequestOutputToDict(pred):
    if not isinstance(pred,RequestOutput):
        return pred
    pred_dict = {
        "request_id":pred.request_id,
        "prompt":pred.prompt,
        "prompt_token_ids":pred.prompt_token_ids,
        "prompt_logprobs":pred.prompt_logprobs,
        "outputs":[{
            "index":pred.outputs[0].index,
            "text":pred.outputs[0].text,
            "token_ids":pred.outputs[0].token_ids,
            "cumulative_logprob":pred.outputs[0].cumulative_logprob,
            "logprobs":pred.outputs[0].logprobs,
            "finish_reason":pred.outputs[0].finish_reason,
            }],
        "finished":pred.finished
    }
    return pred_dict

def calculate_hash(str_array):
    combined_str = ''.join(str_array).encode('utf-8')
    hash_object = hashlib.sha256(combined_str)
    return str(hash_object.hexdigest())


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RequestOutput):
            return None  
        else:
            return super().default(obj)

def changeExcelToJson(filePath:str,type:str):
    finalJsonObjList = []
    xl = pd.ExcelFile(filePath)
    sheet_names = xl.sheet_names
    for sheet_name in sheet_names:
        finalJsonObj = {
            "id":filePath,
            "header":[], # [str]
            "rows":[],    # [str]
            # "data":[]           # [List[Any]]
        }
        if type == 'row_column':
            df = pd.read_excel(filePath, sheet_name=sheet_name)
            for index,row in df.iterrows():
                temp_data_list = []
                skip_first_column = True
                for column_name, cell_value in row.items():
                    if skip_first_column:
                        skip_first_column = False
                        continue
                    temp_data_list.append(cell_value)
                    if column_name not in finalJsonObj['header']:
                        finalJsonObj['header'].append(column_name)
                finalJsonObj['rows'].append(temp_data_list)
            # finalJsonObj['rows'] = df[df.columns[0]].to_list()

        elif type == 'column':
            df = pd.read_excel(filePath, sheet_name=sheet_name)
            for index,row in df.iterrows():
                temp_data_list = []
                for column_name, cell_value in row.items():
                    temp_data_list.append(cell_value)
                    if column_name not in finalJsonObj['header']:
                        finalJsonObj['header'].append(column_name)
                finalJsonObj['rows'].append(temp_data_list)


        elif type == 'row':
            df = pd.read_excel(filePath, sheet_name=sheet_name,header=None)
            finalJsonObj['header'] = df[df.columns[0]].tolist()
            for column in df.columns[1:]:
                finalJsonObj['rows'].append(df[column].tolist())
        else:
            return []
        finalJsonObjList.append(finalJsonObj)
    # print("finalJsonObjList=>\n",finalJsonObjList)
    return finalJsonObjList

def loadKnowledgeByCatch(knowledge_path,elasticIndex,tag):
    
    catchData = []
    catch_flag = False
    for catchDataObj in RootConfig.tempPipeLineKnowledgeCatch:
        if catchDataObj['path'] == knowledge_path or catchDataObj['path'] == elasticIndex:
            catch_flag = True
            catchData = catchDataObj['data']
            
    if catch_flag == False:
        if elasticIndex != "":
            document_store = ElasticsearchDocumentStore(hosts = RootConfig.ES_HOST,index=elasticIndex,basic_auth=(RootConfig.ES_USERNAME,RootConfig.ES_PASSWORD))
            catchData = document_store._search_all_documents()
            catchData = list(map(lambda x:x.to_dict(),catchData))
            catchData = [{**d, 'source': tag} if d['source'] is None else d for d in catchData]
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":elasticIndex,"data":catchData})
            
        
        elif knowledge_path != "":
            catchData = read_server_files(knowledge_path)
            catchData = [{**d, 'source': tag} if d['source'] is None else d for d in catchData]
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})
            
        else:
            catchData = []
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})

    return catchData


def loadStructKnowledgeByCatch(knowledge_path,knowledge_type):
    from kninjllm.llm_retriever.multi_kg.KnowledgeBase.KG_sparse_api import KnowledgeGraphSparse
    catchData = []
    catch_flag = False
    for catchDataObj in RootConfig.tempPipeLineKnowledgeCatch:
        if catchDataObj['path'] == knowledge_path:
            catch_flag = True
            catchData = catchDataObj['data']
            
    if catch_flag == False:
        if knowledge_type == 'kg':
            kg_source_path = ""
            ent_type_path = ""
            ent2id_path = ""
            rel2id_path = ""
            entity_name_path = ""
            for fileName in os.listdir(knowledge_path):
                filePath = os.path.join(knowledge_path,fileName)
                if "triples" in fileName:
                    kg_source_path = filePath
                if "ent_type_ary" in fileName:
                    ent_type_path = filePath
                if "ent2id" in fileName:
                    ent2id_path = filePath
                if "rel2id" in fileName:
                    rel2id_path = filePath
                if "entity_name" in fileName:
                    entity_name_path = filePath
            print("kg_source_path : \t ",kg_source_path)
            print("ent_type_path : \t ",ent_type_path)
            print("ent2id_path : \t ",ent2id_path)
            print("rel2id_path : \t ",rel2id_path)
            print("entity_name_path : \t ",entity_name_path)
            if kg_source_path == "" or ent_type_path == "" or ent2id_path == "" or rel2id_path == "" or entity_name_path == "":
                raise ValueError("KG knowledge error...")
                    
            triples = KnowledgeGraphSparse(triples_path=kg_source_path, ent_type_path=ent_type_path)
            ent2id = _load_pickle_file(ent2id_path)
            id2ent = _reverse_dict(ent2id)
            rel2id = _load_pickle_file(rel2id_path)
            id2rel = _reverse_dict(rel2id)
            entity_name = _load_pickle_file(entity_name_path)
            catchData = [
                {
                    "key":"triples",
                    "value":triples
                },
                {
                    "key":"ent2id",
                    "value":ent2id
                },
                {
                    "key":"id2ent",
                    "value":id2ent
                },
                {
                    "key":"rel2id",
                    "value":rel2id
                },
                {
                    "key":"id2rel",
                    "value":id2rel
                },
                {
                    "key":"entity_name",
                    "value":entity_name
                }
            ]
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})
            
        elif knowledge_type == 'db':
            print(f"reload db ...")
            catchData = []
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})
        
        elif knowledge_type == 'table':
            print(f"reload table ...")
            # catchData = changeExcelToJson(knowledge_path,'column')
            with open(knowledge_path,'r',encoding='utf-8') as f:
                catchData = json.load(f)
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})
            
        elif knowledge_type == "webqsp":
            with open(knowledge_path, "rb") as f:
                cvt_flag_dict, mid_mapping_dict = pickle.load(f)
            catchData = (cvt_flag_dict,mid_mapping_dict)
            RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path,"data":catchData})
            
        else:
            raise ValueError(f"Not know type Struct Knowledge...")

    if len(catchData) == 0:
        raise ValueError("The knowledge quantity of this knowledge source is blank. Please check the knowledge source...")
    
    return catchData


def loadModelByCatch(model_name,model_path):

    
    catchData = None
    catch_flag = False
    for catchDataObj in RootConfig.tempModelCatch:
        if catchDataObj['path'] == model_path:
            catch_flag = True
            catchData = catchDataObj['data']
            
       
    if catch_flag == False:
        if model_name == "selfrag":
            model = LLM(model=model_path,dtype="half", tensor_parallel_size=1)
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            catchData = (model,tokenizer)
            RootConfig.tempModelCatch.append({"path":model_path,"data":catchData})
            model = None
            tokenizer = None
            catchData = None
            
        elif model_name == "llama2":
            llama2_model = LLM(model_path, dtype="half")
            catchData = llama2_model
            RootConfig.tempModelCatch.append({"path":model_path,"data":catchData})
            llama2_model = None
            catchData = None
        
        elif model_name == "baichuan2":
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(model_path)
            catchData = (model,tokenizer)
            RootConfig.tempModelCatch.append({"path":model_path,"data":catchData})
            model = None
            tokenizer = None
            catchData = None
            
        elif model_name == "NED":
            model = Refined.from_pretrained(model_name=model_path,
                                    entity_set="wikidata",
                                    download_files=True,
                                    use_precomputed_descriptions=True)
            catchData = model
            RootConfig.tempModelCatch.append({"path":model_path,"data":catchData})
            model = None
            catchData = None
        
        elif model_name == "wikisp":
            wikisp_model = LLM(model_path, dtype="half")
            catchData = wikisp_model
            RootConfig.tempModelCatch.append({"path":model_path,"data":catchData})
            wikisp_model = None
            catchData = None
        
        else:
            raise ValueError(f"Unsupported model types...")

    result_list = list(filter(lambda x:x['path'] == model_path,RootConfig.tempModelCatch))
    
    if len(result_list) <= 0:
        raise ValueError("Model loading failed, please check...")
    
    return result_list[0]['data']


def read_server_files(path):
    
    def read_one_file(file_path):
        data = []
        if file_path.endswith(".json"):
            with open(file_path, 'r',encoding='utf-8') as file:
                tempData = json.load(file)
                if isinstance(tempData,list):
                    data.extend(tempData) 
                else:
                    data.append(tempData)
        elif file_path.endswith(".jsonl"):
            with open(file_path, 'r',encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line))
        
        elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
            with open(file_path, 'rb') as file:
                try:
                    while True:
                        dict_obj = pickle.load(file)
                        # dict_obj = {key: dict_obj[key] for key in ['id', 'content','source']}
                        data.append(dict_obj)
                except EOFError:
                    pass    
                  
        elif file_path.endswith(".csv") or file_path.endswith(".tsv"):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = [line for line in file]
            data = list(map(lambda x:{"id":x.split("\t")[0],"content":x},lines))
                        
        else:
            print(f"File types not currently supported : {file_path}")

        return data
    
    final_data_list = []
    # is dir
    if os.path.isdir(path):  
        
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            data = read_one_file(file_path)
            final_data_list.extend(data)
    
    # is file
    elif os.path.isfile(path):  
        data = read_one_file(path)
        final_data_list.extend(data)
    else:
        raise ValueError(f"Path is not a folder or file or path does not exist: {path}")

    return final_data_list
    
def EmbeddingByRetriever(documents,retrieverNameList):
    print("utils -> EmbeddingByRetriever ....",retrieverNameList)
    
    paramList = list(map(lambda x:{'id':x['id'],"title":x['content'].split("\t")[2],"content":x['content'].split("\t")[1]},documents))
    
    for retrieverName in retrieverNameList:
        id_data_config = {}
        
        if "contriever" in retrieverName:
            embedding_name = "contriever_embedding"
            allids_list,allembeddings_list = do_contriever_embedding(passages=paramList)
        elif "BGE" in retrieverName:
            embedding_name = "BGE_embedding"
            allids_list,allembeddings_list = do_BGE_embedding(passages=paramList)
        elif "DPR" in retrieverName:
            embedding_name = "DPR_embedding"
            allids_list,allembeddings_list = do_DPR_embedding(passages=paramList)
        elif "E5" in retrieverName:
            embedding_name = "E5_embedding"
            allids_list,allembeddings_list = do_E5_embedding(passages=paramList)
        elif "BERT" in retrieverName:
            embedding_name = "BERT_embedding"
            allids_list,allembeddings_list = do_BERT_embedding(passages=paramList)
        else:
            # continue
            raise ValueError("Unsupported retriever embedding types")
            
        for id,emb in zip(allids_list,allembeddings_list):
            id_data_config[id] = emb.tolist()
            
        for document in documents:
            if len(allids_list) == len(allembeddings_list) == len(documents):
                document[embedding_name] = id_data_config[document['id']]
            else:
                raise ValueError("embedding incorrect length ...")
            
    return documents