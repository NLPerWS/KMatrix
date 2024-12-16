import json
from typing import Any, Dict, List, Optional
import numpy as np
from kninjllm.llm_common.component import component
import os


@component
class Evaluator:

    def __init__(
        self,
    ):
        self.dataSetList = []
        self.final_eva_result = []
        self.index = 0
    
    @component.output_types(final_result=Dict[str, Any])
    def run(
        self,
        dataset_info:Dict[str, Any] = {},
        result : Any = [],
    ):
        if dataset_info != {} and len(dataset_info['dataset_data']) != 0:
            self.dataSetList = dataset_info['dataset_data']
        if isinstance(result,list) and len(result) != 0:
            result = result[0]
        if isinstance(result,dict):
            self.final_eva_result.append(result)
            self.index += 1
            
        if self.index == len(self.dataSetList) and self.index == len(self.final_eva_result):
            print("-----------------------------------Evaluator ---------------------------------------",len(self.dataSetList))
            eva_list = []
            for tempResDoc,datasetObj in zip(self.final_eva_result,self.dataSetList):
                check_flag = 0
                for ans in datasetObj['golden_answers']:
                    if ans in tempResDoc['content']:
                        check_flag = 1
                        break
                eva_list.append({"question":datasetObj['question'],"golden_answers":datasetObj['golden_answers'],"content":tempResDoc['content'],"metric_result":check_flag})
            
            metric_results = list(map(lambda x:x['metric_result'],eva_list))
            
            eva_data = {
                "metric_mean": np.mean(metric_results),
                "eva_list":eva_list
            }
            
            return {"final_result":{"query_obj":{},"eva_result":eva_data,"flag":1}}
        
        else:
            return {"final_result":{"query_obj":self.dataSetList[self.index],"eva_result":{},"flag":0}}
            
            
