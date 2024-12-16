
import json
import os
from typing import Any, Dict, List
from kninjllm.llm_common.component import component
import pandas as pd
from kninjllm.llm_utils.common_utils import calculate_hash

@component
class KGPreprecess:
    def __init__(self):
        pass
    
    
    def execute_one_file(self,filepath):
        finalJsonObjList = []
        if filepath.endswith(".json"):
            with open(filepath,'r',encoding='utf-8') as f:
                datas = json.load(f)
            for data in datas:
                for triples in data['triples']:
                    for i, triple_str in enumerate(triples):
                        triples[i] = triple_str.replace("\t", "   ")
                finalJsonObj = {"id":calculate_hash([str(data['triples'])]),"triples":data['triples'],"source":"知识图谱"}
                finalJsonObjList.append(finalJsonObj)
    
        elif filepath.endswith(".jsonl"):
            with open(filepath,'r',encoding='utf-8') as f:
                for data in f:
                    data = json.loads(data)
                    for triples in data['triples']:
                        for i, triple_str in enumerate(triples):
                            triples[i] = triple_str.replace("\t", "   ")
                    finalJsonObj = {"id":calculate_hash([str(data['triples'])]),"triples":data['triples'],"source":"知识图谱"}
                    finalJsonObjList.append(finalJsonObj)
        else:
            raise ValueError('Unsupported file format')
        return finalJsonObjList
    
    @component.output_types(documents=List[Dict[str, Any]])
    def run(
        self,
        path:Any,
    ):
        if isinstance(path,dict):
            if path['knowledge_elasticIndex'] != '':
                path = path['knowledge_elasticIndex']
            else:
                path = path['knowledge_path']
        elif isinstance(path,str):
            path = path
        else:
            raise ValueError("Preprecess paramter error ...")
        
        finalJsonObjList = []
        if os.path.isdir(path):
            for fileName in os.listdir(path):
                filepath = os.path.join(path,fileName)        
                temp_list = self.execute_one_file(filepath)
                finalJsonObjList.extend(temp_list)
        else:
            finalJsonObjList = self.execute_one_file(path)
        
        return {"documents":finalJsonObjList}
    