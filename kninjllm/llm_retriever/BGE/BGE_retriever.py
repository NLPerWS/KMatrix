import json
import os
from typing import Any, Dict, List, Optional
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import loadKnowledgeByCatch,unset_proxy
from root_config import RootConfig
from kninjllm.llm_retriever.BGE.infer import infer
from kninjllm.llm_retriever.BGE.test_new import test
from kninjllm.llm_retriever.BGE.train import train



@component
class BGE_Retriever:

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
        knowledge_info : Dict[str,Any] = {},
        test_data_info: Dict[str,Any] = {},
        train_data_info : Dict[str,Any] = {},
        dev_data_info : Dict[str,Any] = {},
    ):
        
        if knowledge_info != {}:
            knowledge_path = knowledge_info['knowledge_path']
            knowledge_elasticIndex = knowledge_info['knowledge_elasticIndex']
            knowledge_tag = knowledge_info['knowledge_tag']
            knowledge = loadKnowledgeByCatch(knowledge_path=knowledge_path,elasticIndex=knowledge_elasticIndex,tag=knowledge_tag)
            if len(knowledge) > 0:
                self.searchDataList = knowledge
            if knowledge_path != "":
                self.knowledge_path = knowledge_path
        
        if self.executeType == "infer":
            print("BGE infer ...")
            
            if query_obj == {}:
                return {"final_result": []}
            
            if len(self.searchDataList) == 0 and self.knowledge_path == "":
                return {"final_result": []}
            
            query = query_obj['question']
            
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> BGE_Retriever -> run | Given the search text, return the search content ")
                self.logSaver.writeStrToLog("search input -> : query: "+str(query))
                
            
            print("------------self.model_path---------")
            print(self.model_path)
            
            final_result = infer(
                model=self.model_path,
                input_query=[query],
                passage=self.searchDataList,
                top_k=self.top_k,
            )
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("search returned -> : final_result: "+str(final_result))
            unset_proxy()
            return {"final_result": final_result}
        
        elif self.executeType == "evaluate":
            
            print("------------test_data_info-----------")
            print(test_data_info)
            
            if test_data_info['dataset_path'] == "":
                return {"final_result": {}} 
            
            output_path = RootConfig.root_path + "TEMP_BEG_test_output_" + test_data_info['dataset_path'].replace("/", "_").replace("\\","_").replace(".","")
            print("--------------output_path---------------")
            print(output_path)
            
            metrics = test(model_path=self.model_path,dataset_path=test_data_info['dataset_path'],output_path=output_path)
            
            unset_proxy()
            return {"final_result":metrics} 
        
        elif self.executeType == "train":
            previous_directory = os.getcwd()   
            os.chdir(RootConfig.root_path + "kninjllm/llm_retriever/BGE")
            if train_data_info['dataset_path'] == "":
                return {"final_result": {}} 
            print("train_data_path")
            print(train_data_info['dataset_path'])
            
            out_dir_name = "TEMP_TRAIN_BGE-Reproduce"
            output_model_name = RootConfig.root_path + out_dir_name  
            train_model_path = self.model_path  
            train_data = train_data_info['dataset_path']  
            train(train_data,train_model_path,output_model_name)
            os.chdir(previous_directory)
            unset_proxy()
            return {"final_result":{"content":f"train start,please wait... (please check log in log file {out_dir_name} )","meta":{}}} 

        else:
            raise ValueError(f"obj must be has 'executeType',but your not {self.executeType}")
        