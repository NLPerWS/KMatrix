import json
import sys
import time
from typing import Any, Dict, List
import pandas as pd

from kninjllm.llm_common.document import Document
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import calculate_hash



@component(is_greedy=True)
class UnifiedInterface:

    def __init__(self,
                    knowledge_line_count,
                    max_length = 0,
                    valueList:List[Any] = [],
                    count = 0):
        self.valueList = valueList
        self.count = count
        self.knowledge_line_count = knowledge_line_count
    
    @component.output_types(final_result=Dict[str,Any])
    # def run(self, **kwargs):
    def run(self, value:Any):
        print("-----------------------------UnifiedInterface --------------------------------")
        self.count += 1
        
        if self.count > self.knowledge_line_count:
            raise ValueError("Error in initial parameters of linearizer. Please check...")
        
        self.valueList.append(value)
        
        print("finally UnifiedInterface len:  \n",len(self.valueList))
        
        if self.knowledge_line_count != self.count:
            return {"final_result":{"flag":0,"knowledge":[]}}
        else:
            return {"final_result":{"flag":1,"knowledge":self.valueList}}
        