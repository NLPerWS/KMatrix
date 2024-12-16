from argparse import Namespace
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

import sys
import numpy as np

from kninjllm.llm_utils.common_utils import loadKnowledgeByCatch,unset_proxy
from kninjllm.llm_common.component import component
from transformers import AutoTokenizer

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from kninjllm.llm_retriever.contriever.passage_retrieval import main_infer
from kninjllm.llm_retriever.contriever.test_new import test
from kninjllm.llm_retriever.contriever.train import train

from root_config import RootConfig

@component
class Contriever_Retriever:

    def __init__(
        self,
        top_k = 10,
        model_path:str="",
        executeType:str="",
    ):
        self.top_k = top_k
        self.logSaver = RootConfig.logSaver
        self.executeType = executeType
        
        self.model_path = model_path
        
        self.searchDataList = []
        self.knowledge_path = ""
        
        
    @component.output_types(final_result=List[List[Dict[str,Any]]])
    def run(
        self,
        query_obj: Dict[str,Any] = {},
        query_list: list[Dict[str,Any]] = [],
        knowledge_info : Dict[str,Any] = {},
        test_data_info: Dict[str,Any] = {},
        train_data_info : Dict[str,Any] = {},
        dev_data_info : Dict[str,Any] = {},
    ):
        print("-------------------------------Contriever_Retriever-----------------------------------")
        
        if knowledge_info != {}:
            knowledge_path = knowledge_info['knowledge_path']
            knowledge_elasticIndex = knowledge_info['knowledge_elasticIndex']
            knowledge_tag = knowledge_info['knowledge_tag']
            if knowledge_elasticIndex != "":
                knowledge = loadKnowledgeByCatch(knowledge_path="",elasticIndex=knowledge_elasticIndex,tag=knowledge_tag)
            else:
                knowledge = []
            if len(knowledge) > 0:
                self.searchDataList = knowledge
                for index,know in enumerate(self.searchDataList):
                    if len(know['content'].split("\t")) != 3:
                        know['content'] = know['source']+str(index) + "\t" + know['content'].replace("\t","") + "\t" + "None"
                    know['title'] = know['content'].split("\t")[2]
                    
            if knowledge_path != "":
                self.knowledge_path = knowledge_path
        
        if self.executeType == "infer":
            
            print("Contriever_Retriever infer ...")
            print("final do contriever knowledge len ",len(self.searchDataList))
            print("final do contriever knowledge path ",self.knowledge_path)
            
            if len(self.searchDataList) == 0 and self.knowledge_path == "":
                return {"final_result": []}
            
            if query_obj == {} and query_list == []:
                return {"final_result": []}
            
            if query_list == [] and query_obj != {}:
                query_list = [query_obj]
            
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> Contriever_Retriever -> run | Given the search text, return the search content ")
                self.logSaver.writeStrToLog("search input -> :query_obj: "+str(query_list))

            args = Namespace(dataSetList=query_list,
                             passages=self.searchDataList, 
                             passages_path=self.knowledge_path, 
                             n_docs=self.top_k,
                             validation_workers=32,
                             per_gpu_batch_size=64,
                             save_or_load_index=False,
                             model_name_or_path=self.model_path,
                             no_fp16=False,
                             question_maxlength=512,
                             indexing_batch_size=1000000,
                             projection_size=768,
                             n_subquantizers=0,
                             n_bits=8,
                             lang=None,
                             dataset="none",
                             lowercase=False,
                             normalize_text=False,
                             )
            
            data = main_infer(args)
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("search returned -> : final_result: "+str(data))
                
            unset_proxy()
            return {"final_result": data}
        
        elif "evaluate" in self.executeType:
            
            if test_data_info['dataset_path'] == "":
                return {"final_result": {}} 
            
            output_path = RootConfig.root_path + "TEMP_Contriever_test_output_" + test_data_info['dataset_path'].replace("/", "_").replace("\\","_").replace(".","")
            print("--------------output_path---------------")
            print(output_path)
            
            res = test(model_path=self.model_path,dataset_path=test_data_info['dataset_path'],output_path=output_path)
            unset_proxy()
            return {"final_result":res} 
        
        elif self.executeType == "train":
            previous_directory = os.getcwd()   
            os.chdir(RootConfig.root_path + "kninjllm/llm_retriever/contriever")
            if train_data_info['dataset_path'] == "":
                return {"final_result": {}} 
            print("train_data_path")
            print(train_data_info['dataset_path'])
            out_dir_name = "TEMP_TRAIN_contriever"
            output_model_name = RootConfig.root_path + out_dir_name  
            train_model_path = self.model_path  
            train_data = train_data_info['dataset_path'] 
            train(train_data,train_model_path,output_model_name)
            
            os.chdir(previous_directory)
            unset_proxy()
            return {"final_result":{"content":f"train start,please wait... (please check log in log file {out_dir_name} )","meta":{}}} 

        else:
            raise ValueError(f"obj must be has 'executeType',but your not {self.executeType}")
        