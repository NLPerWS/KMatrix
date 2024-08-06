import json
import os
import time
from typing import Any, Dict, List, Literal
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import calculate_hash
from root_config import RootConfig

@component
class SaveToServer:

    def __init__(
        self,
        savePath
    ):
        self.savePath = savePath

        if self.savePath != "" and not self.savePath.startswith("/"):
            self.savePath = os.path.join(RootConfig.root_path,self.savePath)
        

    @component.output_types(final_result=Dict[str,Any])
    def run(self, documents:List[Dict[str,Any]]):
        if self.savePath == "":
            raise ValueError("savepath is empty")
        
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        print("---------------------save documents-------------------------")
        print(len(documents))
        
        hash_value = calculate_hash(list(map(lambda x:x['id'],documents)))
        tempSaveName = str(hash_value) + ".json"
        saveFlag = True
        for checkName in os.listdir(self.savePath):
            if checkName == tempSaveName:
                saveFlag = False
                break
        final_save_path = os.path.join(self.savePath,tempSaveName)
        if saveFlag:
            with open(final_save_path, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=4)
                
        return {"final_result":f"save to server ok : {final_save_path}"}